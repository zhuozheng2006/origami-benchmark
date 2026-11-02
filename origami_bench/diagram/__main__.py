"""python -m origami_bench.diagram: run a diagram policy against a real shell."""
import argparse, json, shlex, subprocess
from pathlib import Path
from .episode import Episode, DEFAULT_DATA, write_json, check_freeze


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    sub.add_parser('verify-data')
    for name in ['run','oracle','agent']:
        s=sub.add_parser(name);s.add_argument('--task',required=True);s.add_argument('--output',required=True)
        if name=='run': s.add_argument('--actions',required=True);s.add_argument('--agent-kind',default='submitted_actions')
        if name=='agent': s.add_argument('--command',required=True);s.add_argument('--timeout',type=float,default=90);s.add_argument('--agent-id',required=True)
    args=p.parse_args()
    if args.mode=='verify-data': print(check_freeze());return
    out=Path(args.output).resolve();episode=None
    try:
        episode=Episode(args.task,out)
        if args.mode=='oracle':
            actions=episode.private['reference_actions']+[{'op':'finish'}]
            kind='oracle_controller_qualification_NOT_AI_accuracy'
        elif args.mode=='run':
            actions=json.loads(Path(args.actions).read_text());kind=args.agent_kind
        else:
            actions=None;kind='external_agent:'+args.agent_id
        for i in range(12):
            observation=episode.observation()
            if actions is None:
                # Contract: one observation JSON on stdin, ONE action JSON on
                # stdout. Model process gets only the allow-listed observation.
                result=subprocess.run(shlex.split(args.command),input=json.dumps(observation),text=True,capture_output=True,timeout=args.timeout,cwd=DEFAULT_DATA/'public')
                (out/f'agent-{i:03d}.stderr.txt').write_text(result.stderr)
                (out/f'agent-{i:03d}.stdout.txt').write_text(result.stdout)
                if result.returncode: raise RuntimeError('Agent process failed; see stderr artifact')
                try: action=json.loads(result.stdout)
                except json.JSONDecodeError as ex:
                    write_json(out/'agent-error.json',dict(category='model_protocol_error',detail=str(ex)));break
            elif i<len(actions): action=actions[i]
            else: break
            decision=episode.step(action)
            print(json.dumps(dict(task=args.task,decision=i,result=decision),ensure_ascii=False),flush=True)
            if episode.finished or decision.get('category')=='environment_error': break
            # Oracle qualification must expose a failure, not silently retry.
            if actions is not None and decision['status']=='rejected': break
        report=episode.finish(kind)
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    except Exception as ex:
        out.mkdir(parents=True,exist_ok=True)
        write_json(out/'environment-error.json',dict(category='environment_error',detail=type(ex).__name__+': '+str(ex),task_id=args.task))
        raise
    finally:
        if episode is not None: episode.close()


if __name__=='__main__': main()
