"""One portable, accounted harness for reference smoke tests and user agents."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from origami_bench.diagram.accounted_episode import AccountedEpisode
from origami_bench.diagram import layer_policy_v15 as policy
from origami_bench.diagram.layer_episode_v16 import LayerEpisodeV16
from origami_bench.diagram.local_backend import LocalLayerClient

SETTINGS = dict(backend_variant='codim-ipc-directional', barrier_distance=.0002,
                compression_barrier=True, grip_solve_tolerance=.0002,
                grip_solve_metric='rms', newton_tolerance=1e-4)


class ReplayAgent:
    def __init__(self, actions):
        self.actions = iter(actions)

    def request_public(self, text, image_paths):
        return dict(action=next(self.actions, None), input_tokens=0,
                    output_tokens=0, raw_output='scripted replay')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', required=True, help='ID in data/evaluation-catalog.json')
    p.add_argument('--allow-diagnostic',action='store_true',help='Explicitly run tasks whose scoring still needs repair')
    p.add_argument('--output', type=Path, required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--reference', action='store_true', help='Oracle installation check, never a model score')
    mode.add_argument('--actions', type=Path, help='JSON list of all attempted actions; replay, including rejected actions')
    mode.add_argument('--agent', help='Importable module:factory; factory() returns request_public(text,image_paths) agent')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Existing attempts are immutable; choose a new output directory')
    from verify_release import verify
    manifest=verify()
    release_digest=hashlib.sha256((ROOT/'release-manifest.json').read_bytes()).hexdigest()
    catalog=json.loads((ROOT/'data/evaluation-catalog.json').read_text())['tasks']
    if args.task not in catalog:
        raise ValueError('Task is not runnable; see audit/inventory.json for its blockers')
    spec=catalog[args.task]
    if spec['status']!='accepted' and not args.allow_diagnostic:
        raise ValueError('Diagnostic task requires --allow-diagnostic and never contributes to accepted-track scores')
    data = ROOT / 'data' / spec['dataset']
    task = next(t for t in json.loads((data/'public/index.json').read_text())['tasks'] if t['id'] == args.task)
    private = json.loads((data/'private/tasks.json').read_text())[args.task]
    for key in ('goal','initial_goal','checkpoints'):
        if isinstance(private.get(key),str):private[key]=str(ROOT/private[key])
    for page in private.get('pages',[]):
        if not (ROOT/page).is_file():raise FileNotFoundError('Missing source page. Run scripts/prepare_external_sources.py first: '+page)
    if spec.get('requires_source_download'):
        from PIL import Image
        source_specs=json.loads((ROOT/'data/external-qualified/source-manifest.json').read_text())
        expected={f".local/diagram-sources-tiered-v1/{s['id']}/page-{p['page']:03}.png":p for s in source_specs for p in s['pages']}
        for page in private['pages']:
            im=Image.open(ROOT/page).convert('RGB');target=expected[page]
            if list(im.size)!=target['size'] or hashlib.sha256(im.tobytes()).hexdigest()!=target['pixel_sha256']:
                raise ValueError('Source page pixels changed: '+page)
    private['pages']=[str(ROOT/x) for x in private.get('pages',[])]
    decisions=spec['max_attempts']
    if spec['episode']=='step':
        from origami_bench.diagram import line_fold_agent_v9 as active_policy
    else:
        active_policy=policy
    if args.reference:
        actions=[dict(a,op='fold_line') if a.get('op')=='fold' else a for a in private['reference_actions']]
        agent = ReplayAgent(actions + [{'op':'finish'}])
        kind = 'oracle_installation_check'
    elif args.actions:
        actions = json.loads(args.actions.read_text())
        if not isinstance(actions, list) or len(actions) > decisions:
            raise ValueError('Action list exceeds the track attempt budget')
        agent = ReplayAgent(actions)
        kind = 'submitted_action_replay'
    else:
        module, factory = args.agent.split(':')
        agent = getattr(importlib.import_module(module), factory)()
        kind = 'external_agent:' + args.agent
    wrapped = None
    try:
        if spec['episode']=='step':
            from origami_bench.diagram.public_face_episode import PublicFaceEpisode
            ep=PublicFaceEpisode(args.task,args.output.resolve(),data=data,
                backend_factory=lambda mesh:LocalLayerClient(mesh,worker='codim_worker.py',backend_variant='codim-ipc'),max_decisions=decisions)
        elif spec['episode'] in ('document','hard'):
            from origami_bench.diagram.hard_episode import HardEpisode
            from origami_bench.diagram.document_episode import DocumentEpisode
            klass=HardEpisode if spec['episode']=='hard' else DocumentEpisode
            ep=klass(task,private,args.output.resolve(),lambda mesh:LocalLayerClient(mesh,worker='document_worker.py',backend_variant='codim-ipc'),decisions)
            from origami_bench.diagram import document_policy as active_policy
        else:
            ep = LayerEpisodeV16(task, private, args.output.resolve(),
                                lambda mesh: LocalLayerClient(mesh, **SETTINGS),
                                decisions, fixture_mode='guided_directional_10000')
        wrapped = AccountedEpisode(ep, 8192)
        for i in range(decisions):
            result, stop = wrapped.turn(agent, active_policy)
            print(json.dumps(dict(attempt=i+1, result=result)), flush=True)
            if stop:
                break
        # Legacy step episodes did not persist their final state; save it for the common viewer.
        (ep.output/'final-state.json').write_text(json.dumps(ep.backend.state()))
        report = wrapped.finish(kind)
        report['release_track'] = spec['track']
        report['suite']=spec['suite']
        report['policy_version']=active_policy.VERSION
        report['release_version'] = manifest['version']
        report['release_manifest_sha256'] = release_digest
        report['score_eligible'] = kind.startswith('external_agent:') and spec['status']=='accepted'
        (ep.output/'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/'startup-or-runtime-error.json').write_text(json.dumps({
            'status':'environment_error', 'task_id':args.task,
            'error_type':type(error).__name__,
            'detail':'Execution did not complete; inspect local stderr and worker log. No success certified.'
        },indent=2))
        raise
    finally:
        if wrapped:
            wrapped.close()
        if hasattr(agent, 'close'):
            agent.close()


if __name__ == '__main__':
    main()
