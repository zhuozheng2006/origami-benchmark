"""Build the evidence-only validation viewer; no solver or API is used."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'validation-v3'


def build():
    grading=json.loads((ROOT/'audit/grading-v3-validation.json').read_text())
    identity=json.loads((ROOT/'audit/identity-validation-v3.json').read_text())
    names={
        'reference_path':'参考路径', 'global_yaw_xyz_and_height_drift':'允许整体摆放变化和高度漂移',
        'extra_samples_valid_decomposition':'合法动作分解及额外采样', 'no_motion_static_padding':'静止不动并重复填帧',
        'skip_required_state':'跳过必要状态', 'reverse_state_order':'颠倒状态顺序',
        'four_duplicate_checkpoints':'四个完全重复检查点', 'near_checkpoints_with_padded_frames':'相近检查点加静止重复帧',
        'required_paper_rotation_present':'执行必要转纸', 'required_paper_rotation_omitted':'省略必要转纸',
        'mirror_material':'镜像材料位置', 'wrong_mountain_valley':'山谷方向错误', 'wrong_turnover':'翻面错误',
        'scaled_paper':'缩放纸张', 'all_evidence_present':'证据门槛全部满足', 'missing_finish':'缺少结束动作',
        'unknown_physics':'物理有效性未知', 'failed_physics':'物理检查失败',
        'missing_signed_crease_memory':'缺少必要有向折痕记忆', 'environment_fault':'环境故障',
        'empty_path':'空轨迹', 'empty_targets':'空检查点', 'nan_geometry':'非有限几何数据',
        'test_same_adapter_different_models_stay_separate':'相同适配器区分两个模型',
        'test_parameters_and_grading_stay_separate':'不同参数与评分版本分组',
        'test_legacy_missing_identity_excluded':'旧报告身份缺失不参与比较',
        'test_duplicate_run_ids_and_directories_do_not_inflate':'重复证据只计一次',
        'test_coverage_missing_and_startup_failure_visible':'遗漏任务和启动失败可见',
        'test_declared_identity_mismatch_is_not_scored':'声明身份不符拒绝计分',
        'test_credentials_not_copied_from_metadata':'身份元数据不复制密钥',
        'test_returned_model_mismatch_is_recorded':'返回模型差异显式记录',
        'test_suite_preregisters_repeats_and_retains_startup_failures':'预注册重复实验及启动失败',
        'test_environment_failure_stays_in_model_denominator':'环境失败保留在同一模型分母',
        'test_returned_identity_change_blocks_comparison':'返回身份变化阻止直接比较'}
    tests=[dict(name=names.get(c['name'],c['name']),expected=c['expected'],actual=c['actual'],
                passed=c['verified'],kind='synthetic',detail=c.get('details',{}).get('reason','评分组件／证据条件对照，不是真实物理实验'))
           for c in grading['controls']]
    tests += [dict(name=names.get(c['name'],c['name']),expected='断言满足',actual='断言满足' if c['passed'] else '断言失败',
                   passed=c['passed'],kind='synthetic',detail=c.get('detail') or '身份与汇总离线测试，不是模型成绩') for c in identity['cases']]
    runs=[]
    for r in grading['archived_runs']:
        folder=ROOT/r['evidence'];state=json.loads((folder/'final-state.json').read_text())
        spec=json.loads((ROOT/'data/evaluation-catalog.json').read_text())['tasks'][r['task_id']]
        private=json.loads((ROOT/'data'/spec['dataset']/'private/tasks.json').read_text())[r['task_id']]
        a=np.asarray(state['vertices']);b=np.load(ROOT/private['goal'])['vertices']
        ac=a[:,:2].mean(0);bc=b[:,:2].mean(0);u,_,vt=np.linalg.svd((a[:,:2]-ac).T@(b[:,:2]-bc))
        if np.linalg.det(u@vt)<0:u[:,-1]*=-1
        xy=a.copy();xy[:,:2]=(a[:,:2]-ac)@(u@vt)+bc
        aligned=xy.copy();dz=float(np.mean(a[:,2]-b[:,2]));aligned[:,2]-=dz
        recorded=json.loads((folder/'report.json').read_text())
        note='历史记录离线重评分；不是新模型调用或新的物理运行。'
        if 'gpt' in r['run']:
            note+=' 高度差已从形状分中消除；两次展开顺序与图解不同，过程仍只匹配 4/8，因此没有改判整个任务成功。'
        if 'gemini' in r['run']:
            note+=' 旧规则两次重复使用同一帧计转纸；新规则要求真正转动，8 个检查点分别在不同状态匹配。'
        runs.append(dict(id=r['run'],label=r['task_id']+' · '+r['run'],
            old_status=recorded.get('status'),candidate_status='annotation_invalid' if not r['diagram_path']['contract_valid'] else
                'environment_error' if r['environment_error'] else 'passed_scoped' if r['task_success'] else 'not_passed',
            shape_before=r['legacy_shape'],shape_after=r['shape'],z_offset_mm=dz*1000,
            path=r['diagram_path'],old_path=r.get('legacy_diagram_path'),physical_pass=r['physical_pass'],
            memory_pass=r['memory_pass'],finished=r['finished'],note=note,
            mesh=dict(vertices=a.round(7).tolist(),xy_aligned_vertices=xy.round(7).tolist(),
                      aligned_vertices=aligned.round(7).tolist(),reference_vertices=b.round(7).tolist(),faces=state['faces'],units='m')))
    runs.sort(key=lambda r: ('gpt' not in r['id'],'gemini' not in r['id'],r['id']))
    report=dict(version=grading['grading_version'],source_commit='aeea10305b94eb71f1726742f9e31dcb71325080',
        generated_at='2026-11-07',summary=dict(paid_calls=0,new_physics_runs=0,legacy_unchanged=grading['original_evidence_unchanged']),
        tests=tests,runs=runs,identity=dict(before_groups=1,after_groups=identity['demonstrations']['different_models']['output_groups'],
            models=['合成模型 A','合成模型 B'],notes='同一适配器下两模型分开；一次成功加一次环境错误仍计两次，成功率 50%；重复证据只计一次。'),
        contract_notes=['结构只忽略整体平移及平面朝向，不允许镜像、缩放或任意三维旋转。',
          '过程固定初始方向，因此图解规定的后续转纸仍需执行。',
          '相邻必要状态的容差区域必须可分；重复或过近检查点属于题目规范歧义，不能重复得分。',
          '几何对齐只用于评分；实际接触、应变、折痕记忆与结束动作仍独立检查。',
          'O04 仍属诊断题；不更改历史报告、不计入正式排名。'])
    with (OUT/'report.json').open('x') as f:json.dump(report,f,ensure_ascii=False,separators=(',',':'))
    print(f'{len(tests)} controls; {len(runs)} historical runs; viewer data written')


if __name__=='__main__':build()
