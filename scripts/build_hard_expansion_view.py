"""Read-only evidence export for hard-v4/index.html; no inference or simulation.

Up to 100 uniformly sampled frames plus every recorded action endpoint are
exported for display. Original trajectories, scores, failures and ledger rows
are never changed. Source PNGs go only into the ignored .local/hard-v4 cache.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import sys
from urllib.parse import quote, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parents[1]
VERSION = 'hard-expansion-view-v1'
FINAL_GRADING_VERSION = 'precrease-artifact-signed-memory-v4-candidate2'
LEGACY_GRADING_VERSION = 'precrease-artifact-signed-memory-v4-candidate1'
sys.path.insert(0, str(ROOT))
from origami_bench.diagram.document_grading import retained_history


def read(path, default=None, warnings=None):
    path = Path(path)
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        if warnings is not None:
            warnings.append(f'{path.name}: {type(exc).__name__}; result not treated as complete')
        return default


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
    tmp.replace(path)


def safe_url(value):
    return value if isinstance(value, str) and urlparse(value).scheme in ('http', 'https') else None


def local_path(value, artifact_root, source_root):
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [artifact_root / path, source_root / path]
    return next((p for p in candidates if p.is_file()), candidates[0])


def display_pages(candidate, artifact_root, source_root, public_root, warnings):
    pages = (candidate.get('source') or {}).get('pages', candidate.get('pages', [])) or []
    result = []
    safe_id = re.sub(r'[^A-Za-z0-9_.-]', '_', candidate['id'])
    for index, page in enumerate(pages):
        page = {'local_path': page} if isinstance(page, str) else page
        path = local_path(page.get('local_path', page.get('path')), artifact_root, source_root)
        if path is None or not path.is_file():
            warnings.append(f"{candidate['id']}: source page {index+1} missing")
            continue
        if path.suffix.lower() != '.png' or path.read_bytes()[:8] != b'\x89PNG\r\n\x1a\n':
            warnings.append(f"{candidate['id']}: source page {index+1} is not a PNG; not copied")
            continue
        digest = sha(path)
        if page.get('sha256') and page['sha256'] != digest:
            warnings.append(f"{candidate['id']}: source page {index+1} checksum mismatch; not copied")
            continue
        page_num = page.get('pdf_page_one_based', page.get('page', index+1))
        name = f'{index+1:03}-{digest[:16]}.png'
        target = public_root / '.local/hard-v4' / safe_id / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            tmp = target.with_name(target.name + '.tmp')
            shutil.copyfile(path, tmp)
            tmp.replace(target)
        elif sha(target) != digest:
            raise ValueError('Ignored source cache hash mismatch: ' + str(target))
        result.append(dict(label=page.get('label', f'原图第 {page_num} 页'),
                           url='../.local/hard-v4/'+quote(safe_id)+'/'+quote(name),
                           sha256=digest, original_pdf_page=page_num))
    return result


def attempted_actions(folder, warnings):
    raw = read(folder / 'reference-attempts.json', None, warnings)
    if isinstance(raw, list):
        rows = []
        for i, entry in enumerate(raw):
            result = entry.get('result')
            if not isinstance(result, dict):
                result = dict(status=entry.get('status', 'unknown'))
                if entry.get('error'):
                    result.update(category='environment_error', detail=entry['error'])
            rows.append(dict(attempt_id=entry.get('attempt_id', i+1), action=entry.get('action'),
                             result=result, before_frame=entry.get('before_frame'),
                             after_frame=entry.get('after_frame'), seconds=entry.get('seconds'),
                             reference_action_index=entry.get('reference_action_index')))
        return rows
    events = read(folder / 'events.json', [], warnings)
    config = read(folder / 'run-config.json', {}, warnings)
    before = config.get('initial_frame', 0)
    rows = []
    for i, event in enumerate(events):
        after = event.get('frame')
        rows.append(dict(attempt_id=event.get('index', i)+1, action=event.get('action'),
                         result=event.get('result', {}), before_frame=before, after_frame=after))
        if isinstance(after, int):
            before = after
    return rows


def select_frames(total, uniform_limit, endpoints):
    if total < 1 or not 2 <= uniform_limit <= 100:
        raise ValueError('Nonempty trajectory and sampling limit within 2..100 required')
    uniform = set(np.linspace(0, total-1, min(uniform_limit, total), dtype=int).tolist())
    valid = {int(f) for f in endpoints if isinstance(f, (int, np.integer)) and not isinstance(f, bool) and 0 <= int(f) < total}
    return sorted(uniform | valid), sorted(valid)


def retained_crease_overlay(history, selected_frames, vertex_count):
    """Actual retained branch only; never reads reference/target requirements."""
    if not isinstance(history, list) or not history:
        return None
    branch = retained_history(history)
    if not branch:
        return None
    stencils = np.asarray(branch[0].get('hinges'))
    if stencils.ndim != 2 or stencils.shape[1] != 4 or not np.issubdtype(stencils.dtype, np.integer):
        raise ValueError('History needs integer four-vertex hinge stencils')
    if np.any(stencils < 0) or np.any(stencils >= vertex_count):
        raise ValueError('History hinge vertices outside trajectory')
    steps, angles = [], []
    for entry in branch:
        step = entry.get('steps')
        a = np.asarray(entry.get('angles'), float)
        if type(step) is not int or step < 0 or (steps and step < steps[-1]):
            raise ValueError('Retained history must use monotonic integer frame indices')
        if a.shape != (len(stencils),) or not np.isfinite(a).all():
            raise ValueError('History angle dimensions or finiteness invalid')
        if not np.array_equal(np.asarray(entry.get('hinges')), stencils):
            raise ValueError('History hinge topology changed')
        steps.append(step)
        angles.append(a)
    negative, positive = np.zeros(len(stencils), bool), np.zeros(len(stencils), bool)
    frames, cursor = [], 0
    threshold = np.deg2rad(140.)
    for frame in selected_frames:
        while cursor < len(steps) and steps[cursor] <= frame:
            negative |= angles[cursor] < -threshold
            positive |= angles[cursor] > threshold
            cursor += 1
        # A hinge may legitimately have BOTH retained signs; neither is hidden.
        frames.append([[int(i), sign] for sign, mask in ((-1, negative), (1, positive))
                       for i in np.flatnonzero(mask)])
    return dict(hinge_edges=stencils[:, 1:3].tolist(), frames=frames,
                threshold_degrees=140, original_history_records=len(history), retained_history_records=len(branch),
                sign_reference='original_blue_face', valley_sign=-1, mountain_sign=1,
                display_only=True, target_used=False,
                update_policy='Cumulative signed peaks from retained low-level command endpoints with steps <= display frame; rolled-back branches removed.',
                note='按保留分支的低层指令末帧更新；表示曾经折过的方向，不是当前夹角。')


def export_trajectory(folder, attempts, output, uniform_limit, warnings):
    path = folder / 'trajectory.npz'
    if not path.is_file():
        return {}
    try:
        with np.load(path, allow_pickle=False) as saved:
            vertices = saved['vertices']
            faces = saved['faces'] if 'faces' in saved.files else None
        if vertices.ndim != 3 or vertices.shape[-1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
            raise ValueError('Invalid or nonfinite trajectory')
        if faces is None:
            state = read(folder / 'final-state.json') or read(folder / 'initial-state.json') or read(folder / 'preparation-state.json')
            faces = np.asarray(state['faces']) if state else None
        if faces is None:
            raise ValueError('No recorded triangle faces')
        faces = np.asarray(faces)
        if faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer) or faces.min() < 0 or faces.max() >= vertices.shape[1]:
            raise ValueError('Invalid triangle indices')
        endpoints = [f for a in attempts for f in (a.get('before_frame'), a.get('after_frame'))]
        config = read(folder / 'run-config.json', {})
        endpoints.append(config.get('initial_frame'))
        endpoints.extend(r.get('frame') for r in read(folder / 'preparation-records.json', []))
        selected, valid_endpoints = select_frames(len(vertices), uniform_limit, endpoints)
        invalid = [f for f in endpoints if f is not None and (type(f) is not int or not 0 <= f < len(vertices))]
        if invalid:
            warnings.append(folder.name + ': out-of-range action endpoints retained in attempts but excluded from display sampling')
        name = re.sub(r'[^A-Za-z0-9_.-]', '_', folder.name) + '-' + hashlib.sha256(str(folder.resolve()).encode()).hexdigest()[:8] + '.json'
        payload = dict(units='m', vertices=np.round(vertices[selected], 7).tolist(), faces=faces.tolist(),
                       frame_numbers=selected, total_frames=len(vertices), action_endpoint_frames=valid_endpoints,
                       uniform_sample_limit=uniform_limit, display_only=True,
                       original_trajectory_sha256=sha(path), sampling_note='Uniform samples plus all valid action endpoints; never used for scoring.')
        history_path = folder / 'material-history.json'
        if history_path.is_file():
            try:
                overlay = retained_crease_overlay(read(history_path), selected, vertices.shape[1])
                if overlay is not None:
                    overlay['source_history_sha256'] = sha(history_path)
                    payload['crease_history'] = overlay
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                warnings.append(folder.name + ': retained crease overlay unavailable: ' + str(exc))
        atomic_json(output / 'trajectories' / name, payload)
        return dict(trajectory='trajectories/'+quote(name), trajectory_sampled_frames=len(selected),
                    trajectory_total_frames=len(vertices), trajectory_original_sha256=payload['original_trajectory_sha256'])
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        warnings.append(folder.name + ': trajectory export unavailable: ' + type(exc).__name__)
        return dict(trajectory=None, trajectory_error=type(exc).__name__)


def failure_reason(report, attempts, prep_error, completed):
    failures = [a for a in attempts if a.get('result', {}).get('status') in ('rejected', 'environment_error') or
                a.get('result', {}).get('category') == 'environment_error']
    failure = report.get('failure') or report.get('environment_error')
    if isinstance(failure, dict):
        return f"{failure.get('category', failure.get('stage', '中断'))}：{failure.get('detail', failure.get('error_type', '未知'))}"
    if failures:
        last = failures[-1]
        result = last['result']
        message = f"尝试 {last['attempt_id']}：{result.get('category', result.get('status'))}；{result.get('detail', '动作未完成')}。该尝试保留。"
        if result.get('category') == 'capability_limit' and 'Motion exceeds validated per-action integration budget' in str(result.get('detail', '')):
            message += '接口单动作积分预算上限为600帧；这是接口能力边界，不是模型失败。'
        return message
    if prep_error:
        return '准备阶段未完成：' + str(prep_error.get('error', prep_error.get('detail', '未知原因')))
    if not completed:
        return '验收仍在进行或尚未完整封存；没有完整报告前不判通过。'
    if report.get('candidate_reference_pass') is True:
        return '候选参考路径完整，物理及独立几何检查通过；仍待来源标注、重复验收及模型难度校准。'
    if not report.get('annotated_plan_complete'):
        return '参考路径未完整执行（含预算或 finish 未完成）；有效前缀不等于完整通过。'
    return '执行完成，但候选验收检查未全部通过；保留原始报告，未调整门槛。'


def run_role(folder, private):
    kind = private.get('control_kind')
    negative = (private.get('expected_task_score') is False or
                (isinstance(kind, str) and kind.startswith('negative')) or
                folder.name.startswith('negative-'))
    return ('negative_control' if negative else 'positive_reference'), kind


def scorer_record(folder, report, completed, warnings, score_name='precrease-grading.json', expected_version=None):
    score_path = folder / score_name
    score = read(score_path, None, warnings)
    if not isinstance(score, dict):
        return None
    if score.get('task_id') != report.get('task_id') or type(score.get('task_success')) is not bool:
        warnings.append(folder.name + ': task scorer identity/result invalid; not accepted')
        return None
    if expected_version is not None and score.get('grading_version') != expected_version:
        warnings.append(folder.name + ': scoring version does not match selected edition; not accepted')
        return None
    if not completed:
        warnings.append(folder.name + ': task score present before valid completion marker; not accepted')
        return None
    return dict(task_success=score['task_success'], physical_pass=score.get('physical_pass'),
                memory_pass=score.get('memory_pass'), finished=score.get('finished'),
                environment_error=score.get('environment_error'), shape=score.get('shape'),
                grading_version=score.get('grading_version'), grading_contract=score.get('grading_contract'),
                memory_diagnostics=score.get('memory_diagnostics'), source_sha256=sha(score_path), source_file=score_name)


def scoring_acceptance(task_id, runs, summary):
    """Physical replay success never substitutes for task scoring controls."""
    rows = summary.get('tasks', {}) if isinstance(summary, dict) else {}
    row = rows.get(task_id, {}) if isinstance(rows, dict) else next((x for x in rows if x.get('task_id') == task_id), {})
    positives = [r for r in runs if r['run_role'] == 'positive_reference' and
                 r['candidate_reference_pass'] and r.get('task_success') is True and
                 r.get('grading', {}).get('physical_pass') is True and
                 r.get('grading', {}).get('finished') is True and
                 r.get('grading', {}).get('environment_error') is False]
    negatives = [r for r in runs if r['run_role'] == 'negative_control' and
                 r.get('control_expectation_met') is True]
    distinct = {r.get('run_id') for r in positives if r.get('run_id')}
    negative_types = set()
    for r in negatives:
        label = str(r.get('control_kind') or r['id']).lower()
        if 'empty' in label or 'missing_all' in label:
            negative_types.add('empty')
        if 'sign' in label:
            negative_types.add('sign')
    contracts = {r['grading'].get('grading_contract') for r in positives + negatives}
    versions = {r['grading'].get('grading_version') for r in positives + negatives}
    negative_ids = {r.get('run_id') for r in negatives if r.get('run_id')}
    consistent = bool(len(contracts) == 1 and None not in contracts and '' not in contracts and
                      len(versions) == 1 and None not in versions and '' not in versions)
    ready = bool(row.get('scoring_acceptance_pass') is True and len(distinct) >= 2 and len(negative_ids) >= 2 and consistent and
                 any(r['id'].startswith('alternate-') for r in positives) and
                 negative_types >= {'empty', 'sign'})
    return dict(passed=ready, positive_scored_runs=[r['id'] for r in positives],
                negative_rejected_runs=[r['id'] for r in negatives],
                negative_types=sorted(negative_types), consistent_grading_contract=consistent, summary_declares_pass=row.get('scoring_acceptance_pass') is True,
                reason=row.get('reason') or '需要两条独立正参考（含反向移动侧）及真实 empty/sign 负对照一致通过评分验收。')


def run_record(folder, expected, output, uniform_limit, warnings, score_name='precrease-grading.json', expected_version=None, grading_applicable=True):
    report_path = folder / 'report.json'
    report = read(report_path, {}, warnings)
    marker = read(folder / 'qualification-complete.json', {}, warnings)
    complete = bool(report and marker and marker.get('run_id') == report.get('run_id') and
                    marker.get('report_sha256') == sha(report_path))
    attempts = attempted_actions(folder, warnings)
    plan = read(folder / 'qualification-plan.json', {}, warnings)
    private = read(folder / 'candidate-private.json', {}, warnings)
    role, control_kind = run_role(folder, private)
    sequence_pass = bool(complete and report.get('candidate_reference_pass') is True)
    grading = scorer_record(folder, report, complete, warnings, score_name, expected_version) if grading_applicable else None
    comparison = None
    if score_name == 'precrease-grading-v4-final.json' and grading is not None:
        old = scorer_record(folder, report, complete, warnings, 'precrease-grading.json', LEGACY_GRADING_VERSION)
        if old is not None:
            comparison = dict(legacy_version=old['grading_version'], final_version=grading['grading_version'],
                              legacy_task_success=old['task_success'], final_task_success=grading['task_success'],
                              changed=old['task_success'] != grading['task_success'], same_run_id=report.get('run_id'),
                              legacy_source_sha256=old['source_sha256'], final_source_sha256=grading['source_sha256'],
                              scope='Same recorded run, offline scorer comparison; no new model result.')
    task_success = grading['task_success'] if grading is not None else None
    control_met = (task_success is False and grading.get('physical_pass') is True and
                   grading.get('finished') is True and grading.get('environment_error') is False) if grading is not None and role == 'negative_control' else None
    reason = failure_reason(report, attempts, read(folder / 'preparation-error.json', {}, warnings), complete)
    if role == 'negative_control':
        if sequence_pass:
            reason = '该控制动作序列完整，物理与独立几何检查通过。'
        reason = '负对照（预期任务失败）：' + ('评分正确拒绝。' if control_met else '评分错误通过，不能验收该评分规则。' if task_success is True else '任务评分尚未完成；') + '动作序列物理完成不代表题目做对。' + reason
    elif sequence_pass:
        reason = ('正参考物理序列完整；未采用本页的预折任务评分合同。' if not grading_applicable else
                  '正参考物理序列完整；任务评分通过。' if task_success is True else
                  '正参考物理序列完整，但任务评分未通过。' if task_success is False else
                  '正参考物理序列完整；任务评分和正负对照尚未齐全。')
    if comparison is not None:
        old_label = '通过' if comparison['legacy_task_success'] else '未通过'
        new_label = '通过' if comparison['final_task_success'] else '未通过'
        reason += f' 同一记录评分：候选1 {old_label} → 最终候选2 {new_label}。'
    elif not grading_applicable:
        reason += ' 本题未采用预折评分合同，未进入任务评分；不适用最终候选2评分。'
    elif score_name == 'precrease-grading-v4-final.json' and grading is None:
        reason += ' 当前统一等待最终候选2评分；不以旧候选1结果补空。'
    geometric = report.get('geometric_checks', read(folder / 'geometric-checks.json', [], warnings)) or []
    checks = geometric + (report.get('preparation_checks') or [])
    if isinstance(report.get('initial_check'), dict):
        checks.append(report['initial_check'])
    rms = [c['rms_mm'] for c in checks if isinstance(c, dict) and isinstance(c.get('rms_mm'), (int, float)) and not isinstance(c.get('rms_mm'), bool) and np.isfinite(c['rms_mm'])]
    accepted = sum(a.get('result', {}).get('status') == 'accepted' and (a.get('action') or {}).get('op') != 'finish' for a in attempts)
    prep_error = read(folder / 'preparation-error.json', {}, warnings)
    value = dict(id=folder.name, run_id=report.get('run_id', plan.get('run_id')),
                 run_role=role, control_kind=control_kind, expected_task_score=False if role == 'negative_control' else True,
                 physical_sequence_pass=sequence_pass, task_success=task_success, grading=grading,
                 grading_comparison=comparison, selected_grading_version=expected_version if grading_applicable else None,
                 task_scoring_applicable=grading_applicable,
                 task_scoring_status='not_applicable' if not grading_applicable else 'pending' if grading is None else 'passed' if task_success else 'failed',
                 control_expectation_met=control_met, accepted=report.get('accepted_reference_actions', accepted),
                 expected=report.get('expected_reference_actions', plan.get('expected_reference_actions', expected)),
                 finished=bool(complete and report.get('finished')), physical_pass=report.get('physical_pass') if complete else None,
                 candidate_reference_pass=bool(sequence_pass and role == 'positive_reference'),
                 max_rms_mm=max(rms) if rms else None, reason=reason,
                 trajectory=None, attempts=attempts, report_finalized=complete,
                 original_status=report.get('status'), recorded_candidate_reference_pass=report.get('candidate_reference_pass'),
                 score_eligible=False, rejected_attempts=sum(a['result'].get('status') == 'rejected' for a in attempts),
                 preparation_error=prep_error or None, report_sha256=sha(report_path) if report_path.is_file() else None,
                 source_directory=str(folder), qualification_note='Private reference execution, never an AI model score.')
    value.update(export_trajectory(folder, attempts, output, uniform_limit, warnings))
    return value


def ledger_snapshot(path):
    if not path.is_file():
        return {}, '共享费用账本未找到；费用未知，未据此重置预算。'
    db = sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True)
    try:
        cap = db.execute('SELECT ceiling FROM config WHERE id=1').fetchone()[0]
        operating = db.execute('SELECT operating_ceiling FROM budget_policy WHERE id=1').fetchone()[0]
        calls, charge, reserve = db.execute('SELECT COUNT(*),SUM(COALESCE(charge,0)),SUM(CASE WHEN charge IS NULL THEN reserve ELSE 0 END) FROM calls').fetchone()
    finally:
        db.close()
    charge, reserve = float(charge or 0), float(reserve or 0)
    summary = dict(call_count=calls, estimated_charges_cny=charge, unresolved_reserves_cny=reserve,
                   occupied_cny=charge+reserve, authorized_cap_cny=cap, operating_cap_cny=operating,
                   authorized_headroom_cny=cap-charge-reserve, operating_headroom_cny=operating-charge-reserve,
                   read_only=True, actual_provider_payment=False)
    return summary, (f'同一共享账本：估计费用 {charge:.2f} 元＋历史未结算预留 {reserve:.2f} 元＝保守占用 {charge+reserve:.2f} 元；'
                     f'授权上限 {cap:g} 元（余 {cap-charge-reserve:.2f} 元），保护线 {operating:g} 元（余 {operating-charge-reserve:.2f} 元）。这不是实际付款账单，未释放预留。')


def build(catalog_path, artifact_root, runs_root, public_root, output, ledger, uniform_limit=100):
    catalog = read(catalog_path)
    if not isinstance(catalog, dict) or not isinstance(catalog.get('candidates'), list) or not catalog['candidates']:
        raise ValueError('Nonempty candidate-catalog.json with candidates array required')
    candidates = json.loads(json.dumps(catalog['candidates']))
    ids = [c['id'] for c in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate candidate IDs')
    warnings, private_index = [], {}
    for path in sorted(artifact_root.glob('*/private/tasks.json')):
        tasks = read(path, {}, warnings)
        for task_id, private in tasks.items():
            private_index.setdefault(task_id, private)
    final_summary_path = artifact_root / 'precrease-scoring-final/summary.json'
    final_mode = final_summary_path.is_file() or any(runs_root.glob('*/precrease-grading-v4-final.json'))
    summary_path = final_summary_path if final_mode else artifact_root / 'precrease-scoring/summary.json'
    scoring_summary = read(summary_path, {}, warnings)
    expected_version = FINAL_GRADING_VERSION if final_mode else scoring_summary.get('grading_version')
    score_name = 'precrease-grading-v4-final.json' if final_mode else 'precrease-grading.json'
    if final_mode and scoring_summary and scoring_summary.get('grading_version') != FINAL_GRADING_VERSION:
        warnings.append('Final scoring summary version mismatch; formal acceptance withheld')
        scoring_summary = {}
    precrease_tasks = {c['id'] for c in candidates if c.get('scope') == 'real_source_precreasing_subtask'}
    if isinstance(scoring_summary.get('tasks'), dict):
        precrease_tasks.update(scoring_summary['tasks'])
    grouped = {c['id']: [] for c in candidates}
    for folder in sorted(runs_root.iterdir()) if runs_root.is_dir() else []:
        if not folder.is_dir():
            continue
        config = read(folder/'run-config.json', {}, warnings)
        task = read(folder/'candidate-public.json', {}, warnings)
        report = read(folder/'report.json', {}, warnings)
        task_id = task.get('id') or report.get('task_id') or config.get('public_task', {}).get('id')
        if not task_id:
            matches = [i for i in ids if re.search(r'(^|[^A-Za-z0-9])'+re.escape(i)+r'($|[^A-Za-z0-9])', folder.name)]
            task_id = matches[0] if len(matches) == 1 else 'UNREGISTERED'
        if task_id not in grouped:
            grouped[task_id] = []
            candidates.append(dict(id=task_id, title='未匹配候选注册的运行', scope='来源和步骤范围待核对', reason='保留未登记执行，不能计分', source={}, status='unregistered'))
            warnings.append(folder.name + ': run has no matching catalog entry; retained under '+task_id)
        expected = len(private_index[task_id]['reference_actions']) if task_id in private_index and isinstance(private_index[task_id].get('reference_actions'), list) else None
        grouped[task_id].append(run_record(folder, expected, output, uniform_limit, warnings, score_name, expected_version, task_id in precrease_tasks))
    tasks = []
    for c in candidates:
        source = c.get('source') or {}
        runs = grouped[c['id']]
        positives = [r for r in runs if r['run_role'] == 'positive_reference']
        passed = sum(r['candidate_reference_pass'] for r in positives)
        acceptance = scoring_acceptance(c['id'], runs, scoring_summary)
        finalized = sum(r['report_finalized'] for r in runs)
        rejected = any(r['rejected_attempts'] or r['preparation_error'] or (r['report_finalized'] and not r['candidate_reference_pass']) for r in positives)
        if acceptance['passed']:
            label, kind = '评分正负对照验收通过；待难度校准', 'pass'
            for run in positives:
                if run.get('task_success') is True:
                    run['reason'] += ' 本题独立正参考与真实负对照已完成评分验收。'
        elif passed:
            label, kind = f'正参考物理通过 {passed} 次；待评分验收', 'pending'
        elif any(not r['report_finalized'] and not r['preparation_error'] and not r['rejected_attempts'] for r in positives):
            label, kind = '参考执行进行中 / 待封存', 'pending'
        elif positives and rejected:
            label, kind = '参考未通过；失败保留', 'fail'
        else:
            pending_labels = {'unsupported_mesh_angles':'现接口不支持所需折痕；未评分',
                              'semantic_and_interface_pending':'图解语义与接口待核对',
                              'annotation_pending':'待动作标注与物理验收', 'metadata_pending':'来源资料待补齐'}
            label, kind = c.get('status_label', pending_labels.get(c.get('status'), '待参考验收')), 'pending'
        note = c.get('reason', c.get('note', ''))
        skills = c.get('skills') or []
        challenge = c.get('challenge') or ('、'.join(skills) if isinstance(skills, list) else str(skills)) or note or '图解理解与连续操作；难度尚未校准'
        source_note = c.get('source_note') or source.get('note') or ('原图仅保存于本地忽略缓存；PDF/PNG不加入公开Git。来源文档标题：'+str(source.get('title', c.get('title', '未知'))))
        scope_labels = {'single_sheet_component':'单张纸的部件折叠', 'real_source_precreasing_subtask':'真实图解的预折子任务',
                        'single_sheet_complete_model_candidate':'单张纸完整作品候选', 'real_source_partial_model_candidate':'真实图解的作品中段候选'}
        scope_kind = c.get('scope', '')
        scope_label = scope_labels.get(scope_kind, scope_kind or '范围待明确')
        step_range = c.get('printed_steps')
        scope_text = scope_label + ('；印刷步骤：' + str(step_range) if step_range else '')
        tasks.append(dict(id=c['id'], title=c.get('title', c['id']), scope=scope_text,
                          challenge=challenge, status_label=label, status_type=kind, source_url=safe_url(source.get('download_url')) or safe_url(source.get('page_url')),
                          source_note=source_note, pages=display_pages(c, artifact_root, SOURCE, public_root, warnings),
                          runs=runs, note=note, source_group=c.get('source_group'), catalog_status=c.get('status'),
                          printed_steps=c.get('printed_steps'), formal_score_eligible=False, scoring_acceptance=acceptance,
                          candidate_scoring_validated=acceptance['passed'], positive_reference_count=len(positives),
                          negative_control_count=sum(r['run_role']=='negative_control' for r in runs), complete_reports=finalized))
    try:
        costs, cost_text = ledger_snapshot(ledger)
    except (sqlite3.Error, OSError, TypeError) as exc:
        costs, cost_text = {}, '共享账本读取暂不可用；费用未知，未调整或重置预算。'
        warnings.append('ledger read unavailable: '+type(exc).__name__)
    runs = [r for t in tasks for r in t['runs']]
    counts = dict(candidates=len(tasks), distinct_source_groups=len({t['source_group'] for t in tasks if t['source_group']}),
                  runs=len(runs), finalized_runs=sum(r['report_finalized'] for r in runs),
                  passed_reference_runs=sum(r['candidate_reference_pass'] for r in runs),
                  failed_finalized_runs=sum(r['run_role']=='positive_reference' and r['report_finalized'] and not r['candidate_reference_pass'] for r in runs),
                  negative_control_runs=sum(r['run_role']=='negative_control' for r in runs),
                  correct_negative_controls=sum(r.get('control_expectation_met') is True for r in runs),
                  incorrect_negative_controls=sum(r['run_role']=='negative_control' and r.get('task_success') is True for r in runs),
                  scoring_validated_candidates=sum(t['candidate_scoring_validated'] for t in tasks),
                  compared_grading_runs=sum(r.get('grading_comparison') is not None for r in runs),
                  changed_grading_runs=sum(bool(r.get('grading_comparison', {}).get('changed')) for r in runs if r.get('grading_comparison')),
                  pending_runs=sum(not r['report_finalized'] for r in runs))
    edition_note = (f"当前统一使用最终候选2评分；同一记录候选1→2对照 {counts['compared_grading_runs']} 条，任务通过结论改变 {counts['changed_grading_runs']} 条。候选2新增折边拓扑、资格物理签名及初态一致性检查，未放宽物理阈值。尚缺最终评分的记录显示待评分，旧版不补入新分母。" if final_mode else '当前仅有候选1评分；最终候选2尚未产生。')
    payload = dict(schema=VERSION, created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   overview=f"{counts['candidates']} 道候选，{counts['runs']} 条真实执行记录，其中 {counts['negative_control_runs']} 条负对照；{counts['passed_reference_runs']} 条正参考物理序列通过，{counts['scoring_validated_candidates']} 道完成评分与正负对照验收。{counts['failed_finalized_runs']} 条正参考未通过，{counts['pending_runs']} 条进行中或待封存。负对照不计可用题；物理完整不等于任务得分通过。",
                   grading=edition_note+' 先验收物理动作，再由所选版本的评分文件判定题目是否做对。候选可用必须两条独立正参考（含反向移动侧）评分通过、真实 empty/sign 负对照被正确拒绝，并有验收汇总明确确认。qualifier 的 candidate_reference_pass 只证明提交的动作序列完整，负对照也可能满足它；本页单独显示，不计为完成题目。轨迹采样不参与评分。',
                   ledger=cost_text, footer='本页为只读证据导出；不会调用模型、运行物理、修订旧得分或清除失败。来源图片仅在本地缓存，公开复现需按原站准备来源。',
                   counts=counts, selected_grading=dict(version=expected_version, score_file=score_name,
                       summary_file=str(summary_path), summary_sha256=sha(summary_path) if summary_path.is_file() else None,
                       legacy_results_preserved=True, mixed_versions_allowed=False),
                   ledger_snapshot=costs, catalog_sha256=sha(catalog_path), warnings=warnings, tasks=tasks)
    atomic_json(output/'report.json', payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, default=SOURCE/'artifacts/hard-expansion-20261107')
    parser.add_argument('--catalog', type=Path)
    parser.add_argument('--runs', type=Path, default=SOURCE/'outputs/hard-expansion-20261107/runs')
    parser.add_argument('--public-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--ledger', type=Path, default=SOURCE/'outputs/multimodel-small-v1/costs.sqlite3')
    parser.add_argument('--frames', type=int, default=100, help='Uniform sample count, 2..100; every action endpoint is added')
    args = parser.parse_args()
    if not 2 <= args.frames <= 100:
        parser.error('--frames must be within 2..100; action endpoints are additional')
    public_root = args.public_root.resolve()
    output = (args.output or public_root/'hard-v4').resolve()
    if output.parent != public_root:
        parser.error('--output must be a direct child of --public-root for relative local-image URLs')
    catalog = args.catalog or args.artifact_root/'candidate-catalog.json'
    payload = build(catalog, args.artifact_root.resolve(), args.runs.resolve(), public_root, output, args.ledger.resolve(), args.frames)
    print(json.dumps(dict(output=str(output/'report.json'), **payload['counts'], warnings=payload['warnings']), ensure_ascii=False))


if __name__ == '__main__':
    main()
