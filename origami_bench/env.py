"""Transactional adapter around the pinned MIT FoldingAgent geometry engine."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor/foldingagent/src'))
from origami import Origami

REVISION = 'f6d4fc6aaf4dfa2c82a8b5f7274c198498f4f122'
CAPABILITIES = dict(flat_fold=True, layer_state=True, continuous_collision=False,
                    thin_shell=False, grasp=False, sink=False, crease_plasticity=False)

def plain(value):
    return json.loads(json.dumps(value, allow_nan=False))

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

class InvalidAction(ValueError):
    pass

class UnsupportedAction(InvalidAction):
    pass

class Environment:
    """Coordinates: initial square [0,1]^2; action semantics follow upstream.

    side=+/-1 is the upstream fold-side selector, NOT a universal M/V sign.
    Only flat endpoint geometry is checked. No continuous physical validation.
    """
    def __init__(self):
        self.paper = Origami(front_color='#e7a75d', back_color='#edf2f6')
        self.history = []
        self.trace = []

    def state(self):
        return plain(self.paper.export())

    def observe(self):
        return dict(backend='foldingagent-flat', revision=REVISION,
                    capabilities=CAPABILITIES.copy(), state=self.state())

    def snapshot(self):
        # Includes the full internal crease/history data, not only the visible mesh.
        return copy.deepcopy((self.paper, self.history, self.trace))

    def restore(self, snapshot):
        self.paper, self.history, self.trace = copy.deepcopy(snapshot)

    def step(self, action):
        a = self._validate(action)
        before = copy.deepcopy(self.paper)
        before_hash = digest(self.state())
        try:
            op = a['op']
            if op == 'fold': self.paper.fold(tuple(a['edge']), a['side'])
            elif op == 'unfold': self.paper.unfold(tuple(a['edge']))
            elif op == 'add_vertex': self.paper.add_vertex(tuple(a['edge']), a['position'])
            elif op == 'rotate': self.paper.rotate(a['degrees'])
            elif op == 'flip': self.paper.flip(a['axis'])
            elif op == 'undo':
                if not self.history: raise InvalidAction('Nothing to undo')
                self.paper = copy.deepcopy(self.history[-1])
            state = self.state()
        except Exception:
            self.paper = before
            raise
        if op == 'undo': self.history.pop()
        else: self.history.append(before)
        self.trace.append(dict(action=a, before=before_hash, after=digest(state)))
        return self.observe()

    @staticmethod
    def _validate(action):
        if not isinstance(action, dict): raise InvalidAction('Action must be an object')
        fields = dict(fold={'edge','side'}, unfold={'edge'}, add_vertex={'edge','position'},
                      rotate={'degrees'}, flip={'axis'}, undo=set())
        op = action.get('op')
        if not isinstance(op, str) or op not in fields:
            raise UnsupportedAction(f'Unsupported operation: {op!r}')
        if set(action) != fields[op] | {'op'}: raise InvalidAction('Unexpected or missing fields')
        a = plain(action)
        if 'edge' in a:
            edge = a['edge']
            if not isinstance(edge,list) or len(edge)!=2 or any(type(v)!=int or v<1 for v in edge) or edge[0]==edge[1]:
                raise InvalidAction('edge must contain two distinct positive vertex IDs')
        if 'side' in a and (type(a['side']) != int or a['side'] not in (-1,1)):
            raise InvalidAction('side must be -1 or 1')
        for key in ('position','degrees'):
            if key in a and (type(a[key]) not in (int,float) or not math.isfinite(a[key])):
                raise InvalidAction(f'{key} must be finite')
        if op=='add_vertex' and not 0<a['position']<1: raise InvalidAction('position must be in (0,1)')
        if op=='flip' and a['axis'] not in ('x','y'): raise InvalidAction('axis must be x or y')
        return a

    def render(self, path, debug=False):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.paper.plot(show=False, save_path=str(path), debug=debug)

    def save_trace(self, path):
        data = dict(schema_version=1, backend='foldingagent-flat', revision=REVISION,
                    initial='unit_square', steps=self.trace)
        Path(path).write_text(json.dumps(data, indent=2), encoding='utf-8')

    @classmethod
    def replay(cls, path):
        data = json.loads(Path(path).read_text())
        if (data.get('schema_version'),data.get('backend'),data.get('revision'),data.get('initial')) != (1,'foldingagent-flat',REVISION,'unit_square'):
            raise ValueError('Unsupported trace schema/backend/revision/initial state')
        env = cls()
        for i, entry in enumerate(data['steps']):
            if digest(env.state()) != entry['before']: raise ValueError(f'Pre-state mismatch at {i}')
            env.step(entry['action'])
            if digest(env.state()) != entry['after']: raise ValueError(f'Replay diverged at {i}')
        return env

