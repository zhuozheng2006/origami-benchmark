"""Prepared physical states, 3-view depth observations, and v15 cross-seam controller.

Preparation is actual forward simulation. Only past commands are public; future
reference actions and goals remain private. All agents use AccountedEpisode.
"""
from pathlib import Path
import copy
import json
import shutil
import time
import numpy as np
from .attempt_ledger import atomic_json
from .depth_renderer_v15 import render_public_v15
from .hard_episode import HardEpisode
from .layer_controller_v16 import LayerControllerV16, VERSION
from .layer_grading_v15 import band_crease_memory_check, ordered_checkpoints


class LayerEpisodeV16(HardEpisode):
    def __init__(self, task, private, output, backend_factory, max_decisions=40, fixture_mode='guided_directional_10000'):
        self.task = copy.deepcopy(task)
        self.private = copy.deepcopy(private)
        root = Path(__file__).resolve().parents[2]
        self.private['pages'] = [
            str(root / p) if not Path(p).is_absolute() else str(Path(p))
            for p in self.private.get('pages', [])
        ]
        for key in ('goal', 'initial_goal', 'checkpoints'):
            if self.private.get(key) and not Path(self.private[key]).is_absolute():
                self.private[key] = str(root / self.private[key])
        self.output = Path(output)
        if self.output.exists():
            raise FileExistsError('Existing attempts are immutable; use a new run ID')
        self.output.mkdir(parents=True)
        self.events = []
        self.actions = []
        self.finished = False
        self.closed = False
        self.max_decisions = max_decisions
        self.started = time.monotonic()
        self.page = 1
        self.crop = [0, 0, 1, 1]
        self.initialization = []
        root = Path(__file__).resolve().parents[2]
        for name in [
            'layer_controller',
            'layer_controller_v15',
            'layer_remote',
            'layer_runtime_remote',
            'layer_episode',
            'layer_episode_v15',
            'layer_reference',
            'layer_grading_v15',
            'depth_renderer_v15',
        ]:
            src = Path(__file__).with_name(name + '.py')
            if src.exists():
                shutil.copy2(src, self.output / (name + '.py'))
        for source in ['origami_bench/layer_env.py', 'origami_bench/control_geometry.py', 'scripts/layer_worker.py']:
            shutil.copy2(root / source, self.output / Path(source).name)
        self.backend = backend_factory(private['mesh'])
        self.controller = LayerControllerV16(self.backend, fixture_mode)
        atomic_json(
            self.output / 'run-config.json',
            dict(
                public_task=self.task,
                controller_version=VERSION,
                backend=self.backend.configuration,
                max_attempts=max_decisions,
                fixture_mode=fixture_mode,
                preparation_complete=False,
                measurement_tool=False,
            ),
        )
        try:
            for cmd in private.get('initialization', []) + private.get('preparation', []):
                self.initialization.append(self.controller.execute(cmd))
                atomic_json(self.output / 'preparation-records.json', self.initialization)
                atomic_json(self.output / 'preparation-state.json', self.backend.state())
            self.controller.undo_floor = len(self.controller.records)
            self.initial_frame = self.backend.state()['steps']
            self.initial_state = copy.deepcopy(self.backend.state())
            atomic_json(self.output / 'initial-state.json', self.initial_state)
            initial_shape = None
            if self.private.get('initial_goal'):
                expected = np.load(self.private['initial_goal'])['vertices']
                error = np.linalg.norm(np.array(self.initial_state['vertices']) - expected, axis=1)
                initial_shape = dict(rms_mm=float(np.sqrt(np.mean(error ** 2)) * 1000.0), max_mm=float(error.max() * 1000.0))
                if initial_shape['rms_mm'] >= 1.0 or initial_shape['max_mm'] >= 3.0:
                    raise RuntimeError('Prepared initial state failed geometric acceptance')
            atomic_json(
                self.output / 'run-config.json',
                dict(
                    public_task=self.task,
                    controller_version=VERSION,
                    backend=self.backend.configuration,
                    initialization=self.initialization,
                    initial_frame=self.initial_frame,
                    initial_shape=initial_shape,
                    max_attempts=max_decisions,
                    measurement_tool=False,
                    automatic_panel=False,
                    explicit_finish_phase=False,
                    fixture_mode=self.controller.fixture_mode,
                    preparation_charged_to_agent=False,
                    preparation_complete=True,
                ),
            )
        except Exception as ex:
            atomic_json(self.output / 'preparation-error.json', dict(error=type(ex).__name__ + ': ' + str(ex)))
            try:
                saved = self.backend.save()
                for name in ['trajectory.npz', 'metrics.json', 'actions.json', 'model-config.json', 'material-history.json']:
                    shutil.copy2(saved / name, self.output / name)
                if (saved / 'rejected-candidates').exists():
                    shutil.copytree(saved / 'rejected-candidates', self.output / 'rejected-candidates')
            finally:
                self.close()
            raise

    def observation(self):
        state = self.backend.state()
        image = self.output / f'observation-{len(self.events):03}.png'
        render_public_v15(state, image)
        return dict(
            task={k: self.task[k] for k in ('id', 'instruction', 'title', 'page_count')},
            document=dict(page=self.page, page_count=self.task['page_count'], crop=self.crop),
            diagram=self.private['pages'][self.page - 1],
            current_image=str(image),
            current=dict(state_hash=state['hash']),
            accepted_actions=copy.deepcopy(self.actions),
            previous_action_result=self.events[-1]['result'] if self.events else None,
            max_decisions=self.max_decisions,
            preparation_actions=[r['action'] for r in self.initialization],
            active_folds=[
                dict(
                    fold_id=r['fold_id'],
                    created_by=self.controller.records[r['record_index']]['action'],
                    remaining_angle_deg=abs(r['degrees']),
                )
                for r in self.controller.fold_stack
            ],
        )

    def step(self, action):
        result = super().step(action)
        self.actions = [r['action'] for r in self.controller.records[self.controller.undo_floor:]]
        return result

    def finish(self, agent_kind):
        atomic_json(self.output / 'rejected-endpoints.json', self.controller.rejected_endpoints)
        r = super().finish(agent_kind)
        state = json.loads((self.output / 'final-state.json').read_text())
        render_public_v15(state, self.output / 'final.png')
        history = json.loads((self.output / 'material-history.json').read_text())
        band_pass = band_crease_memory_check(history, self.private.get('required_band_creases', []))
        r['band_memory_pass'] = bool(band_pass)
        r.update(
            initial_frame=self.initial_frame,
            preparation_operations=len(self.initialization),
            controller_version=VERSION,
            scope='Finite-thickness deformable shell with ideal point fixtures; continuous contact and 1% stretch/compression acceptance. Prepared state is physically replayed. No real finger model.',
        )
        r['goal_success'] = bool(r['task_success'] and band_pass)
        if self.private.get('checkpoints'):
            r['diagram_path'] = ordered_checkpoints(
                np.load(self.output / 'trajectory.npz')['vertices'],
                np.load(self.private['checkpoints'])['vertices'],
                self.initial_frame,
            )
            r['task_success'] = bool(r['goal_success'] and r['diagram_path']['passed'])
        else:
            r['task_success'] = bool(r['goal_success'])
        if r['status'] != 'environment_error':
            r['status'] = 'passed_scoped' if r['task_success'] else ('reference_unscored' if not self.private.get('goal') else 'not_passed')
        r['grading_version'] = 'prepared-shape-ordered-path-and-band-memory-v16-1'
        atomic_json(self.output / 'report.json', r)
        return r
