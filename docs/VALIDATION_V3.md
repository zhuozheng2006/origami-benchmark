# 模型身份与评分候选验证（2026-11-07）

本地分支 `work/public-v3-validation` 基于 public-v2；旧标签和报告保留。没有扩题、没有付费模型调用、没有新仿真。默认仍用 legacy-public-v2 评分；结构 / 过程新合同必须显式选择。O04 仍是诊断任务，没有升级为正式评分集。

## 身份与比较

新报告记录 requested model、服务商、参数、输出预算、adapter 源码哈希、提示词文件哈希、发行清单哈希、返回 model、run UUID 和 repeat ID。身份为供应商 / 调用者声明，不是底层权重认证。

- 共用 adapter 的两个模型不再合并；参数或评分版本不同也分组。
- 环境失败保留在所请求模型的分母中，不因缺少 returned model 被拆走。
- 同一文件 / UUID 的重复输入不增加分数；缺身份的旧记录保留，但不能混入新的可比结果。
- `run_suite.py` 在执行前生成 `experiment-plan.json`。汇总需传 `--plan` 才能核对完整覆盖；无 plan 仅为描述性统计。
- 新的自定义 adapter 实现无副作用 `describe_identity()`，或通过 `--identity identity.json` 提供 requested_model / provider / parameters / output_cap（8192）。如实例提供 metadata，必须与声明一致。不会额外创建模型实例进行预注册。

```bash
# 以下是使用方法，本次未运行模型。
python scripts/run_suite.py --suite layers --agent examples.openai_compatible:create --repeats 2 --output outputs/new-comparison
python scripts/summarize_public.py outputs/new-comparison/* --plan outputs/new-comparison/experiment-plan.json
```

## 新评分合同

`structure-and-diagram-motion-v3-candidate1` 仅适用于已有准备态的 layer 任务：

1. 最终结构允许整体 XYZ 平移和正常平面 yaw；不允许镜像、缩放或任意三维旋转，不改材料顶点身份。
2. 过程只在准备态拟合一次 yaw，后续保持固定；每帧消除整体平移。因此高度漂移不改变折叠结构分，图解中的后续转纸仍可辨识。
3. 相邻必要目标（包括初态到首个目标）必须可分：RMS 距离 > 2 mm 或最大距离 > 6 mm。否则其 <1 mm RMS 且 <3 mm 最大误差接受区域可能重叠，明确拒绝为合同歧义。不是仅把游标改成下一帧。
4. 顺序匹配使用严格递增帧；静止帧的复制无法补足可分目标。保留原形状精度、应变、夹持与接触门槛。新增非有限 / 缺失物理数值拒绝检查。
5. 最终结构、必要过程、物理证据、折痕记忆、显式结束分别报告；总判定为这些条件的合取。

这是一份新的、范围明确的任务合同，不是事后替旧版改分。逐状态整体平移不参与结构判分；若新题明确要求板面绝对位置，应使用其他显式合同。它也不认证任意层序差异，尤其不保证小于现有几何容差的错误层序都能被识别。

## 实际记录结果

- O01–O03、L01、两条 O04 参考轨迹：新合同均通过。
- O04 GPT：旧形状 RMS 6.8293 mm → 新结构 RMS 0.1821 mm；但两次 unfold 的折痕 ID 2/3 次序与参考相反，必要过程依旧只有 4/8，**总判定仍未通过**。
- O04 Gemini：8/8 通过。旧匹配帧 `[1071,1432,1791,1791,2151,2517,2691,2691]` 中的转纸被同帧重复计入；新匹配帧为 `[1070,1431,1791,1978,2151,2514,2691,2878]`，转纸实际发生后才匹配。
- O04 Claude：环境错误保持环境错误，不解释为折纸能力失败。
- 第一条 O04 参考原报告为 reference_unscored，不应描述成模型失败或声称旧评分全部通过。

以机器记录中的确切帧号为准。`audit/grading-v3-validation.json` 保存 26 个正负对照、9 条历史记录对比和证据哈希；`audit/identity-validation-v3.json` 保存 11 项身份 / 汇总验证。合成对照不是物理仿真，重评分也不是新模型实验。

## 复现与查看

```bash
python scripts/verify_release.py
python -m unittest discover -s tests -p 'test_*.py' -v
python scripts/validate_model_identity.py --output outputs/identity-new.json
python scripts/validate_grading_v3.py --output outputs/grading-new.json
python scripts/rescore_structural_v3.py --task O04 --run examples/grading-evidence-v3/model-gpt-6-astra-O04-01 --output outputs/O04-v3-new.json
python -m http.server 8773 --bind 127.0.0.1
# 浏览器访问 http://127.0.0.1:8773/validation-v3/
```

这些验证命令不调用模型或启动原生模拟器。输出路径须为新的，原记录不会被覆盖。

未来要在新的物理运行中试用评分，可给 `evaluate_public.py` 或 layer-only 的 `run_suite.py` 增加 `--grading-contract structural-v3`。新目录保留 `report-legacy.json` 和 `candidate-grading.json`；其他轨道在调用模型或模拟器前明确拒绝。此入口本轮进行了离线集成测试，未运行新的原生物理路径。

可视化页面展示三维终态叠加、旧 / 新检查点帧号及全部对照。它只消费独立 JSON 报告，不参与评分。旧 public-v2 仍可按标签原样复现。

## 本轮完成检查

34 项自动测试全部通过（10 项原公开测试、11 项身份测试、11 项评分测试、2 项入口集成测试）。37 项展示对照符合预期；浏览器实际检查了 GPT / Gemini 切换、三角网格显示与检查点逐项对比。所有历史证据与旧标签保留。
