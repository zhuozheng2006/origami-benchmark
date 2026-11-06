"""Portable subprocess transport; the native worker and acceptance rules are unchanged."""
import os
from pathlib import Path
import selectors
import subprocess
import sys

from .layer_runtime_remote import RuntimeLayerRemoteClient

ROOT = Path(__file__).resolve().parents[2]


class LocalLayerClient(RuntimeLayerRemoteClient):
    def __init__(self, mesh, worker='layer_worker.py', **settings):
        if worker not in ('layer_worker.py','codim_worker.py','document_worker.py'):
            raise ValueError('Unknown worker')
        self.on_frame = None
        self.trace = []
        self.mesh = mesh
        self.buffer = b''
        self.poisoned = False
        self.closed = False
        logs = ROOT / 'outputs/transport'
        logs.mkdir(parents=True, exist_ok=True)
        self.log = (logs / 'worker.log').open('a')
        env = dict(os.environ, OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS', '4'))
        self.proc = subprocess.Popen(
            [os.environ.get('ORIGAMI_PYTHON', sys.executable), str(ROOT / 'scripts' / worker)],
            cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            bufsize=0, env=env,
        )
        self.sel = selectors.DefaultSelector()
        self.sel.register(self.proc.stdout, selectors.EVENT_READ)
        try:
            reply = self._request(dict(mesh=mesh, **settings))
            self.material = reply['material']
            self.configuration = reply['configuration']
            self.committed_state = reply['state']
            self.backend_variant = settings.get('backend_variant', 'codim-ipc-directional')
        except BaseException:
            self.close()
            raise

    def save(self):
        folder = Path(self._request({'op': 'save'})['path']).resolve()
        if not folder.is_relative_to((ROOT / 'outputs/interactive-codim').resolve()):
            raise ValueError('Unexpected worker artifact path')
        return folder

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if not self.poisoned and self.proc.poll() is None:
                self._request({'op': 'close'})
                self.proc.wait(timeout=5)
        except Exception:
            pass
        finally:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()
            self.sel.close()
            self.log.close()
