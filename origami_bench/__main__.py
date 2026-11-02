import argparse
import json
from pathlib import Path
from .env import Environment, ROOT
from .evaluate import compare_states

def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='command',required=True)
    demo=sub.add_parser('demo'); demo.add_argument('--output',default='outputs/smoke')
    replay=sub.add_parser('replay'); replay.add_argument('trace'); replay.add_argument('--png',default='outputs/replay.png')
    run=sub.add_parser('run'); run.add_argument('actions'); run.add_argument('--output',default='outputs/run')
    args=parser.parse_args()
    if args.command=='replay':
        env=Environment.replay(args.trace); env.render(args.png); print('Replay verified'); return
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    if args.command=='run':
        actions=json.loads(Path(args.actions).read_text())
        env=Environment(); env.render(out/'000.png',debug=True)
        for i,a in enumerate(actions,1): env.step(a); env.render(out/f'{i:03}.png',debug=True)
        env.save_trace(out/'trace.json'); print(json.dumps(env.observe(),indent=2)); return
    reports=[]
    for path in sorted((ROOT/'tasks').glob('*.json')):
        task=json.loads(path.read_text()); env=Environment()
        folder=out/task['id']; folder.mkdir(exist_ok=True)
        env.render(folder/'000.png',debug=True)
        for i,action in enumerate(task['reference_actions'],1):
            env.step(action); env.render(folder/f'{i:03}.png',debug=True)
        env.save_trace(folder/'trace.json')
        replayed=Environment.replay(folder/'trace.json')
        report=compare_states(replayed.state(),env.state())
        report.update(task=task['id'],replay_verified=True,kind='reference_smoke_not_model_evaluation')
        reports.append(report)
        (folder/'reference_state.json').write_text(json.dumps(env.state(),indent=2))
    (out/'report.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps(reports,indent=2))

if __name__=='__main__': main()

