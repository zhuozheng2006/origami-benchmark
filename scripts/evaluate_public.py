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
    p.add_argument('--task', required=True, choices=['O01', 'O02', 'O03', 'O04'])
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
    data = ROOT / 'data/release-v1'
    task = next(t for t in json.loads((data/'public/index.json').read_text())['tasks'] if t['id'] == args.task)
    private = json.loads((data/'private/tasks.json').read_text())[args.task]
    if args.reference:
        agent = ReplayAgent(private['reference_actions'] + [{'op':'finish'}])
        kind = 'oracle_installation_check'
    elif args.actions:
        actions = json.loads(args.actions.read_text())
        if not isinstance(actions, list) or len(actions) > 40:
            raise ValueError('Expected at most 40 actions including failed attempts')
        agent = ReplayAgent(actions)
        kind = 'submitted_action_replay'
    else:
        module, factory = args.agent.split(':')
        agent = getattr(importlib.import_module(module), factory)()
        kind = 'external_agent:' + args.agent
    wrapped = None
    try:
        ep = LayerEpisodeV16(task, private, args.output.resolve(),
                            lambda mesh: LocalLayerClient(mesh, **SETTINGS),
                            40, fixture_mode='guided_directional_10000')
        wrapped = AccountedEpisode(ep, 8192)
        for i in range(40):
            result, stop = wrapped.turn(agent, policy)
            print(json.dumps(dict(attempt=i+1, result=result)), flush=True)
            if stop:
                break
        report = wrapped.finish(kind)
        report['release_track'] = task['release_track']
        report['release_version'] = manifest['version']
        report['release_manifest_sha256'] = release_digest
        report['score_eligible'] = kind.startswith('external_agent:') and task['release_track']=='core'
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
