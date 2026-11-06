"""Explicit SSH CPU transport for the public harness; inference stays local.

The remote subprocess receives only simulator actions. API credentials and the
shared cost ledger never enter this transport. Default execution remains local.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import selectors
import shlex
import subprocess
import uuid

from .local_backend import LocalLayerClient, ROOT
from .layer_runtime_remote import RuntimeLayerRemoteClient
from .layer_protocol import signature

VERSION = 'public-ssh-cpu-v1'
WORKERS = {'layer_worker.py', 'codim_worker.py', 'document_worker.py'}
SSH_OPTIONS = ['-o', 'ClearAllForwardings=yes', '-o', 'BatchMode=yes',
               '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3']

SOURCE_CHECK = """import sys,json,hashlib
from pathlib import Path
root=Path(sys.argv[1]).resolve();expected=json.load(sys.stdin);bad=[]
for name,digest in expected.items():
 p=root/name
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:bad.append(name)
print(json.dumps({'ok':not bad,'mismatches':bad,'checked':len(expected)}))
sys.exit(0 if not bad else 2)
"""

OWNED_CLEANUP = """import sys,os,signal
from pathlib import Path
pid=int(sys.argv[1]);folder=sys.argv[2];worker=sys.argv[3];p=Path('/proc')/str(pid)
if p.exists():
 cmd=(p/'cmdline').read_bytes().split(b'\\0')
 if ('scripts/'+worker).encode() in cmd and str((p/'fd/1').resolve())==folder+'/solver.log':os.kill(pid,signal.SIGTERM)
"""


def _absolute(value, name):
    if not isinstance(value, str) or not value.startswith('/') or '..' in PurePosixPath(value).parts or '\n' in value or '\x00' in value:
        raise ValueError(name + ' must be an explicit absolute path without parent traversal')
    return str(PurePosixPath(value))


def settings_from_environment(env=None):
    env = os.environ if env is None else env
    host = env.get('ORIGAMI_SSH_HOST', '')
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*', host):
        raise ValueError('ORIGAMI_SSH_HOST must be an explicit SSH host alias')
    group = int(env.get('ORIGAMI_CPU_GROUP', '4'))
    if group not in (4, 5, 6):
        raise ValueError('This project reserves only CPU groups 4, 5, 6')
    return dict(host=host, remote_root=_absolute(env.get('ORIGAMI_REMOTE_ROOT'), 'ORIGAMI_REMOTE_ROOT'),
                python=_absolute(env.get('ORIGAMI_REMOTE_PYTHON'), 'ORIGAMI_REMOTE_PYTHON'), cpu_group=group)


def transport_environment(env=None):
    """Do not even inherit task credentials/ledger settings into SSH children."""
    env = os.environ if env is None else env
    return {k:v for k,v in env.items() if not k.startswith('ORIGAMI_')}


def source_manifest(mesh, worker):
    if worker not in WORKERS or not re.fullmatch('[a-z0-9_]+', mesh):
        raise ValueError('Unknown worker or invalid mesh name')
    files = list((ROOT/'origami_bench').glob('*.py'))
    files += [ROOT/'scripts'/worker, ROOT/'scripts/run_bounded.py']
    candidates = [ROOT/'tasks'/kind/(mesh+'.json') for kind in ['crease_meshes', 'multifold_meshes']]
    meshes = [p for p in candidates if p.is_file()]
    if len(meshes) != 1:
        raise ValueError('A unique public mesh is required')
    files += meshes
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def worker_command(remote_root, python, cpu_group, worker):
    if worker not in WORKERS or cpu_group not in (4, 5, 6):
        raise ValueError('Invalid worker or CPU group')
    # flock serializes this project's simultaneous jobs even across local clients.
    # CUDA visibility is changed only in this CPU subprocess, never the SSH session.
    command = ['flock', '-n', f'/tmp/origami-public-cpu-group-{cpu_group}.lock',
               'env', 'CUDA_VISIBLE_DEVICES=', python, 'scripts/run_bounded.py',
               '--cpus', '4', '--group', str(cpu_group), python, 'scripts/'+worker]
    return 'cd ' + shlex.quote(remote_root) + ' && exec ' + shlex.join(command)


class SSHLayerClient(RuntimeLayerRemoteClient):
    def __init__(self, mesh, worker='layer_worker.py', *, host, remote_root, python, cpu_group=4, **settings):
        config = settings_from_environment(dict(ORIGAMI_SSH_HOST=host, ORIGAMI_REMOTE_ROOT=remote_root,
            ORIGAMI_REMOTE_PYTHON=python, ORIGAMI_CPU_GROUP=str(cpu_group)))
        self.host, self.remote_root, self.python, self.cpu_group = (config[k] for k in ['host','remote_root','python','cpu_group'])
        self.worker = worker
        self.transport_env = transport_environment()
        self.on_frame = None; self.trace = []; self.mesh = mesh; self.buffer = b''
        self.poisoned = False; self.closed = False; self.proc = None; self.sel = None
        self._saved_folder = None
        self.total_timeout = int(os.environ.get('ORIGAMI_TOTAL_TIMEOUT', '3600'))
        self.idle_timeout = int(os.environ.get('ORIGAMI_IDLE_TIMEOUT', '600'))
        if min(self.total_timeout, self.idle_timeout) <= 0:
            raise ValueError('Positive physical transport timeouts required')
        self.source_hashes = source_manifest(mesh, worker)
        logs = ROOT/'outputs/transport'; logs.mkdir(parents=True, exist_ok=True)
        self.log_path = logs/(uuid.uuid4().hex+'-ssh-worker.log')
        self.log = self.log_path.open('x')
        try:
            check_command = shlex.join([self.python, '-c', SOURCE_CHECK, self.remote_root])
            verified = subprocess.run(['ssh', *SSH_OPTIONS, self.host, check_command],
                input=json.dumps(self.source_hashes), text=True, capture_output=True, timeout=60, env=self.transport_env)
            if verified.returncode != 0:
                raise RuntimeError('Remote physics source verification failed: '+verified.stdout[-2000:])
            result = json.loads(verified.stdout)
            if result.get('ok') is not True or result.get('checked') != len(self.source_hashes):
                raise RuntimeError('Remote source verification did not complete')
            self.proc = subprocess.Popen(['ssh', *SSH_OPTIONS, self.host,
                worker_command(self.remote_root, self.python, self.cpu_group, worker)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, bufsize=0, env=self.transport_env)
            self.sel = selectors.DefaultSelector(); self.sel.register(self.proc.stdout, selectors.EVENT_READ)
            reply = self._request(dict(mesh=mesh, **settings))
            self.material = reply['material']; self.configuration = reply['configuration']
            self.committed_state = reply['state']; self.backend_variant = settings.get('backend_variant','codim-ipc-directional')
            self.configuration.update(transport_variant=VERSION, transport_host=self.host,
                transport_remote_root=self.remote_root, transport_remote_python=self.python,
                transport_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                transport_total_timeout_seconds=self.total_timeout, transport_idle_timeout_seconds=self.idle_timeout,
                remote_source_hashes=self.source_hashes, cpu_group=self.cpu_group, inference_location='local')
            expected_path = os.environ.get('ORIGAMI_PHYSICS_SIGNATURE_FILE')
            if expected_path:
                expected = json.loads(Path(expected_path).read_text())
                if signature(self.configuration) != expected:
                    raise RuntimeError('Frozen reference physics signature differs; no model call may start')
                self.configuration['expected_physics_signature_sha256'] = hashlib.sha256(Path(expected_path).read_bytes()).hexdigest()
        except BaseException:
            self.close()
            raise

    def _artifact_name(self, folder):
        prefix = self.remote_root+'/outputs/interactive-codim/'
        if not isinstance(folder, str) or not folder.startswith(prefix) or not re.fullmatch('[a-f0-9]{32}', folder[len(prefix):]):
            raise ValueError('Unexpected remote worker artifact path')
        return folder[len(prefix):]

    def save(self):
        remote = self._request({'op':'save'})['path']
        name = self._artifact_name(remote)
        target = ROOT/'outputs/interactive-codim'/name
        if self._saved_folder is None:
            target.mkdir(parents=True, exist_ok=False)
            self._saved_folder = target
        elif target != self._saved_folder:
            raise ValueError('Worker changed artifact identity')
        # Worker directories contain simulation evidence only; no model credentials.
        subprocess.run(['rsync','-a','--checksum','-e', shlex.join(['ssh',*SSH_OPTIONS]),
            self.host+':'+shlex.quote(remote+'/'),str(target)+'/'],check=True,timeout=180,env=self.transport_env)
        return target

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.proc is not None and not self.poisoned and self.proc.poll() is None:
                try:
                    self._request({'op':'close'}); self.proc.wait(timeout=10)
                except Exception:
                    self.poisoned = True
            if self.proc is not None and self.poisoned:
                cfg = getattr(self,'configuration',{}); pid = cfg.get('worker_process_id'); folder = cfg.get('worker_artifact_folder')
                if type(pid) is int:
                    try:
                        self._artifact_name(folder)
                        cmd = shlex.join([self.python,'-c',OWNED_CLEANUP,str(pid),folder,self.worker])
                        subprocess.run(['ssh',*SSH_OPTIONS,self.host,cmd],stdout=subprocess.DEVNULL,stderr=self.log,timeout=15,env=self.transport_env)
                    except Exception:
                        pass
            if self.proc is not None and self.proc.poll() is None:
                self.proc.terminate()
                try:self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:self.proc.kill(); self.proc.wait(timeout=5)
        finally:
            if self.sel is not None:self.sel.close()
            self.log.close()


def configured_backend(mesh, worker='layer_worker.py', **settings):
    mode = os.environ.get('ORIGAMI_BACKEND_MODE', 'local')
    if mode == 'local':
        return LocalLayerClient(mesh, worker=worker, **settings)
    if mode == 'ssh':
        return SSHLayerClient(mesh, worker=worker, **settings_from_environment(), **settings)
    raise ValueError('ORIGAMI_BACKEND_MODE must be local or ssh')
