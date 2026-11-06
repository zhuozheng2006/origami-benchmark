# Origami Benchmark — public evaluation release

**新增 v4：两道独立真实图源的长预折任务（H102、H103），共 131 道已验收任务。** 使用同一控制器与提示词；每题 28 个参考动作加 finish，两题分别通过两条独立正参考和真实负对照；H102 另通过公开入口整合检查。它们采用单列的 `precrease-artifact-v4` 合同，测有向预折历史，不测严格动作顺序；困难等级仍待模型校准，本轮未调用付费模型。[新增题说明](docs/HARD_EXPANSION_V4.md) · [原图与三维验收](hard-v4/index.html)。

**当前分支包含 2026-11-07 的身份修复、可选评分验证及小规模校准协议；public-v2 标签保持原样。** 默认旧评分不变；新合同使用 `--grading-contract structural-v3`。先前离线验证无模型调用，见 [验证说明](docs/VALIDATION_V3.md) 和 [评分对照](validation-v3/index.html)。后续三模型 O01–O04 校准使用 [冻结协议](docs/PILOT_V3_PROTOCOL.md)；[新旧结果页面](study-v3/index.html) 将缺失结果、环境错误、诊断题与正式分数分开显示。

图解驱动、具有有限厚度和接触检查的交互式折纸评测。公开任务、网格、初始状态、参考动作、目标状态、评分器、原生求解器源码和统一运行入口，使用者无需访问作者服务器。

本版按历史验收证据恢复全部可用题，不再仅限三道多层题。全量清单：**[可搜索分类表](audit/inventory.html)** · [机器可读清单](audit/inventory.json) · [运行注册表](data/evaluation-catalog.json)。GitHub 不执行 HTML，下载后直接打开或用 `python -m http.server` 查看。

| 轨道 / `--suite` | 可正式运行题数 | 说明 |
|---|---:|---|
| `basics` | 7 | 原 8 道基础题，D006 参考未通过 |
| `holdout1` | 8 | 第一批历史泛化题，现为公开校准数据 |
| `holdout2` | 7 | 第二批历史泛化题，J001 参考未通过 |
| `small` | 99 | 原 120 题中 75 道 test + 24 道 development；21 道参考未通过 |
| `layers` | 3 | O01–O03；O04 为额外诊断题 |
| `external` | 5 | E01、E02、M03、D04、L01；需本地获取原站图解 |
| `precrease-hard` | 2 | H102、H103；独立原图长预折，显式使用 `--grading-contract precrease-artifact-v4` |

合计 **124 道自带图解 + 7 道原站获取已验证的题 = 131 道已验收可运行题**。另公开 **13 道可运行诊断题**（O04 和 12 道 U 系列）及 **2 道素材获取待修复题**（H03、D05，原站 HTTP 403，支持用户自备匹配 PDF）。默认套件不运行诊断/待修复题；它们不计入正式分数。当前资格以运行注册表为准，原始冻结数据中的候选状态文字保留不改。图解署名见 [DATA_ATTRIBUTION.md](docs/DATA_ATTRIBUTION.md)。

**这些数字是任务实例，不是独立作品数量，也不是新做的模型泛化实验。** 多批数据共享图风、折叠家族或原始题源。早期逐步裁图接口与后期整页自主读图接口分轨报告；不能将它们或旧版本模型分数合成一个排名；development 与 test 也分别汇总。

## 1. 安装

物理运行支持 **Linux x86_64、AVX2/FMA/BMI2、Python 3.10**，使用 CPU，不需要 GPU。建议至少 16 GB 内存用于编译；默认编译并行度 2。macOS/Windows 可查看记录和重新计算已有证据分数；原生物理运行应使用满足条件的 Linux 主机或虚拟机。

```bash
git clone https://github.com/zhuozheng2006/origami-benchmark.git
cd origami-benchmark
# Debian/Ubuntu 的系统构建依赖（Python 3.10 需预先安装）
sudo apt-get update
sudo apt-get install -y build-essential cmake libeigen3-dev libboost-all-dev freeglut3-dev python3-dev
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-public.txt
python scripts/verify_release.py
bash scripts/install_native.sh       # layers / ultra / L01
bash scripts/install_native_base.sh  # basics / holdout1 / holdout2 / small / 整页原站题
```

原生源码及 C++ 依赖随仓库打包，无需下载不固定的上游 HEAD。安装脚本保留已有安装，不静默覆盖。Python 解释器切换后应明确重建对应扩展。源码许可证和来源见 [native/NOTICE.md](native/NOTICE.md)。

## 2. 不调用模型，先验证安装

```bash
python -m unittest discover -s tests -p test_public_release.py -v
python scripts/rescore_evidence.py --task O01 --run examples/reference-evidence/O01
OMP_NUM_THREADS=4 python scripts/evaluate_public.py --task O01 --reference --output outputs/reference-O01
```

第二条只对附带的历史记录重新计算分数，**不等于新仿真**。第三条在本机从平纸开始，执行准备动作，再执行参考路径并重新检查物理有效性和目标，最终写入 `report.json`。所有运行目录必须是新的；旧记录不可覆盖。`--reference` 是读取答案的安装检查，不计为模型成绩。

## 3. 接入自己的视觉模型

统一接口为 `request_public(text, image_paths)`，返回字典中的 `action` 为一个 JSON 动作。框架传入整页图解/所选裁剪、当前纸张视图、完整公开尝试历史和剩余预算；不向模型请求传入参考动作、目标网格和评分检查点。可实现 `your_agent.py` 中的 `create()` 工厂，然后运行：

```bash
python scripts/evaluate_public.py --task O01 --agent your_agent:create --output outputs/my-model-O01
```

也附带不自动重试的 OpenAI-compatible HTTPS 接口示例。以下变量都由使用者自行设置；不要将密钥提交到仓库。`read -s` 适用于 Bash：

```bash
export ORIGAMI_API_BASE='https://YOUR-PROVIDER/v1'
export ORIGAMI_MODEL='YOUR-VISION-MODEL-ID'
read -rsp 'API key: ' ORIGAMI_API_KEY; export ORIGAMI_API_KEY; echo
export ORIGAMI_COST_LIMIT=10
export ORIGAMI_INPUT_RATE=10
export ORIGAMI_OUTPUT_RATE=50
export ORIGAMI_COST_LEDGER="$PWD/outputs/my-costs.sqlite3"
export ORIGAMI_API_LOGDIR="$PWD/outputs/api-my-model-O01"
python scripts/evaluate_public.py --task O01 --agent examples.openai_compatible:create --output outputs/my-model-O01
```

费用变量使用同一货币单位；示例数字不是任何服务商报价。输入/输出单价应设为每百万 token 的保守上限；共享账本跨任务复用，日志目录每次新建。该估算不能保证供应商实际扣费上限，尤其是服务商不遵守输出 token 上限时，应同时配置供应商账户限额。现有账本有 90 单位内部运营上限，配置总限额应不超过 90。模型必须支持图像与标准 Chat Completions JSON 响应；不承诺兼容所有代理平台。

若供应商要求 `max_completion_tokens`，设置 `ORIGAMI_OUTPUT_PARAMETER=max_completion_tokens`；默认使用 `max_tokens`。此发布验收没有进行新的付费 API 调用，自定义代理通道已用无 API 的负对照验证。

逐步图解轨道每题最多 **8 次尝试**，整页/多层轨道 **40 次尝试**；本公开比较版本每次输出预算 **8192 token**。失败、查看图解、撤销、结束都消耗尝试；撤销不能删除历史或返还成本。模型必须自己发出 `finish`。同一轨道使用固定接口与提示词，不允许按题切换策略。逐步轨道保留原有“当前面板 + 结束复核”机制；没有结束面板的单步原图使用通用单图适配器，明确告知模型没有结果图，整页轨道不提供下一步提示。

在新的运行名称下批量测试多层轨道的三道题（不要与前面的 O01 示例复用输出目录）：

```bash
for task in O01 O02 O03; do
  export ORIGAMI_API_LOGDIR="$PWD/outputs/api-suite1-$task"
  python scripts/evaluate_public.py --task "$task" --agent examples.openai_compatible:create --output "outputs/suite1-$task" || break
done
python scripts/summarize_public.py outputs/suite1-O01 outputs/suite1-O02 outputs/suite1-O03
```

## 批量运行恢复后的历史题集

```bash
# 列出默认可运行题，确认规模（small 默认含 development 与 test）
python scripts/run_suite.py --suite small --split test --list
# 用参考动作检验安装；不产生模型分数/费用
python scripts/run_suite.py --suite basics --reference --output outputs/basic-reference
# 在同一固定接口下评估自己的模型；每题独立目录，失败也保留；suite 输出目录必须全新
python scripts/run_suite.py --suite small --split test --agent your_agent:create --output outputs/small-model
python scripts/summarize_public.py outputs/small-model/*
# 复杂题只作为诊断，不能混入正式分数
python scripts/evaluate_public.py --task U001 --allow-diagnostic --reference --output outputs/U001-reference
```

旧参考动作中的内部 `fold` 仅在参考模式下按通用规则转成公开 `fold_line` 请求，再经过同一个动作校验与预算通道；没有按题生成答案的分支。公开模型接口不提供测量工具。新的 8192-token 上限/统一历史机制与早年部分模型试验不同，所以历史结果仅作证据，不冒充本次新评测。

## 外部图解的独立获取

不在仓库再分发第三方页面。使用独立渲染环境（避免改变模拟器依赖）：

```bash
python3 -m venv .venv-sources
.venv-sources/bin/python -m pip install -r requirements-sources.txt
.venv-sources/bin/python scripts/prepare_external_sources.py
python scripts/run_suite.py --suite external --reference --output outputs/external-reference
```

工具校验原 PDF 的 SHA256 和每页解码像素，模型运行前再次检查页面像素。四个可下载题源覆盖五道正式任务。T028/T030 当前返回 403，会跳过并说明原因；H03/D05 不进入默认套件。如使用者已有合法获取的相同 PDF，可放在 `SOURCE/source.pdf` 结构中用 `--pdf-root` 校验，再用 `--allow-diagnostic` 单独运行。不要绕过原站权限，也不要用不同图像悄悄替代。

## 4. 评分、汇总与可视化

```bash
python scripts/summarize_public.py outputs/my-model-O01 outputs/my-model-O02 outputs/my-model-O03
python scripts/view_public_run.py --run outputs/my-model-O01 --output outputs/my-model-O01/replay.html
```

打开生成的 `replay.html`，拖动旋转、滚轮缩放、滑动查看轨迹。附带 [D007 单图任务回放](examples/D007-replay.html)、[L01 多层长路径回放](examples/L01-replay.html) 和 [O01 历史参考回放](examples/O01-replay.html)。

public-v2 的 129 道正式任务都有该版成功参考重放；135 份完成报告（含修复前失败与负对照）均经独立重评分核对。完整证据和首次 L01 超时记录见 [发布验证](docs/PUBLIC_RELEASE_V2_VALIDATION.md)。例如，无需原生仿真即可重算附带的逐步任务：

```bash
python scripts/rescore_evidence.py --task D007 --run examples/reference-evidence-v2/followup/final-D007
```

历史整页/多层轨道成功要求（新预折轨道见独立 v4 合同）：显式结束、没有环境错误、所有保留物理步的接触检查通过、拉伸/压缩低于 1%、夹持误差低于 0.5 mm、最终几何 RMS < 1 mm 且最大误差 < 3 mm、折痕记忆与有序几何检查点通过。逐步轨道还按每步语义和对应检查点评分，保持历史容差；它不使用多层折痕记忆评分。模型失败与环境故障必须分别报告；当前汇总同时给出包含环境故障的成功率。批量入口保存每题日志及 suite-attempts.jsonl；启动失败也生成明确的环境错误报告，不能因缺报告从分母消失。报告每道题、重复次数、模型/供应商版本与 release-manifest 哈希，不能只挑成功运行。

已有动作列表可用 `--actions attempts.json` 在本机重新执行；列表须包括失败尝试且不能超过对应轨道预算。它是动作重放，不计为新模型成绩。证据重评分相信输入记录，不能用来认证不可信投稿；公开答案的设置适用于可复现自评，不是防作弊的隐藏测试平台。

## 5. 公开范围与局限

- 参考动作和评分目标位于各数据集的 `private/` 目录。`private` 是历史目录名，表示运行时不进入模型观察，**不表示仓库内保密**。参评模型不得读取它们；本地 Python 插件没有操作系统级隔离，需自律或另行隔离模型进程。
- 初始状态由准备动作实际仿真得到，不把网格直接传送到目标；`initial_goal` 只用于验收。
- 保留已有几何/物理验收规则，没有为跑通而降低阈值。旧默认合同仍保留其姿态/相近检查点限制；可选 structural-v3 已修复高度摆放和重复检查点问题，O04 仍为诊断题。新预折轨道不使用任意参考动作顺序作通过门槛。
- 模拟使用理想分布式夹持和变形薄壳，不是真实机器人手指；当前接口不支持任意折线、任意沉折、曲面折叠或所有现代折法。
- 旧源码和历史文件仅供研究追溯；新入口是 `scripts/evaluate_public.py`，旧的远程脚本不代表本发行版入口。历史结果不自动等于当前版本的模型成绩。
- 不含密钥、供应商个人账本或第三方书页；因缺失可再分发图解的外部题需本地获取素材，失败获取会明确阻止运行。

验证边界与本次检查结果见 [docs/PUBLIC_RELEASE_V2_VALIDATION.md](docs/PUBLIC_RELEASE_V2_VALIDATION.md)。

## v4 长预折轨道

先按 [v4 说明](docs/HARD_EXPANSION_V4.md) 在独立渲染环境获取两份原图，然后运行：

```bash
python scripts/run_suite.py --suite precrease-hard --reference --grading-contract precrease-artifact-v4 --output outputs/precrease-reference
# 评测自己的模型：将 --reference 换为 --agent your_agent:create --identity identity.json
```

每题40次总尝试。原始失败、接口限制、两版候选评分与全部验收证据分开保存，未把未支持操作计入模型分数。
