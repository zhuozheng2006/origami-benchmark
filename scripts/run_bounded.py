"""Limit only this experiment's CPU affinity and library thread pools."""
import os,sys,argparse
p=argparse.ArgumentParser();p.add_argument('--cpus',type=int,default=4);p.add_argument('--group',type=int,default=0);p.add_argument('command',nargs=argparse.REMAINDER);args=p.parse_args()
if args.cpus<1 or args.group<0 or not args.command:raise ValueError('positive CPU count and a command are required')
available=sorted(os.sched_getaffinity(0));start=args.cpus*args.group;selected=available[start:start+args.cpus]
if len(selected)!=args.cpus:raise ValueError('requested CPU group is outside the inherited affinity mask')
os.sched_setaffinity(0,set(selected));os.environ.update(OMP_NUM_THREADS=str(args.cpus),OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
print('Experiment CPU affinity:',selected,'OMP threads:',args.cpus,file=sys.stderr,flush=True);os.execvp(args.command[0],args.command)
