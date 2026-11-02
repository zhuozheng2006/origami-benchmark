# Origami Benchmark — public evaluation release

图解驱动、具有有限厚度和接触检查的交互式折纸评测。公开任务、网格、初始状态、参考动作、目标状态、评分器、原生求解器源码和统一运行入口，使用者无需访问作者服务器。

**核心集合：O01、O02、O03，共三道已具有参考路径验收证据的开发校准题。** O04 随包提供，但因放置高度和过程评分存在争议，仅作为诊断题，不进入核心平均分。历史目录中的其他候选不是本版正式题目。不要将本集合称为百题基准或独立泛化测试集。

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
bash scripts/install_native.sh
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

统一接口为 `request_public(text, image_paths)`，返回字典中的 `action` 为一个 JSON 动作。框架传入整页图解/所选裁剪、当前纸张三视图、完整公开尝试历史和剩余预算；不向模型请求传入参考动作、目标网格和评分检查点。可实现 `your_agent.py` 中的 `create()` 工厂，然后运行：

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

每题最多 **40 次尝试、每次输出预算 8192 token**。失败、查看图解、撤销、结束都消耗尝试；撤销不能删除历史或返还成本。模型必须自己发出 `finish`。O02/O03 使用相同接口与提示词，不允许按题切换策略。

在新的运行名称下批量测试三道核心题（不要与前面的 O01 示例复用输出目录）：

```bash
for task in O01 O02 O03; do
  export ORIGAMI_API_LOGDIR="$PWD/outputs/api-suite1-$task"
  python scripts/evaluate_public.py --task "$task" --agent examples.openai_compatible:create --output "outputs/suite1-$task" || break
done
python scripts/summarize_public.py outputs/suite1-O01 outputs/suite1-O02 outputs/suite1-O03
```

## 4. 评分、汇总与可视化

```bash
python scripts/summarize_public.py outputs/my-model-O01 outputs/my-model-O02 outputs/my-model-O03
python scripts/view_public_run.py --run outputs/my-model-O01 --output outputs/my-model-O01/replay.html
```

打开生成的 `replay.html`，拖动旋转、滚轮缩放、滑动查看轨迹。也可直接打开附带的 [O01 历史参考回放](examples/O01-replay.html)。

核心成功要求：显式结束、没有环境错误、所有保留物理步的接触检查通过、拉伸/压缩低于 1%、夹持误差低于 0.5 mm、最终几何 RMS < 1 mm 且最大误差 < 3 mm、折痕记忆与有序几何检查点通过。模型失败与环境故障必须分别报告；当前汇总同时给出包含环境故障的成功率。报告每道题、重复次数、模型/供应商版本与 release-manifest 哈希，不能只挑成功运行。

已有动作列表可用 `--actions attempts.json` 在本机重新执行；列表须包括失败尝试且最多 40 项。它是动作重放，不计为新模型成绩。证据重评分相信输入记录，不能用来认证不可信投稿；公开答案的设置适用于可复现自评，不是防作弊的隐藏测试平台。

## 5. 公开范围与局限

- 参考动作和评分目标位于 `data/release-v1/private/`。`private` 是历史目录名，表示运行时不进入模型观察，**不表示仓库内保密**。参评模型不得读取它们；本地 Python 插件没有操作系统级隔离，需自律或另行隔离模型进程。
- 初始状态由准备动作实际仿真得到，不把网格直接传送到目标；`initial_goal` 只用于验收。
- 保留已有几何/物理验收规则，没有为跑通而降低阈值。O04 的高度对齐问题尚未解决；有序检查点允许同一帧匹配相近状态，不能声称严格验证了每一条图解操作。
- 模拟使用理想分布式夹持和变形薄壳，不是真实机器人手指；当前接口不支持任意折线、任意沉折、曲面折叠或所有现代折法。
- 旧源码和历史文件仅供研究追溯；新入口是 `scripts/evaluate_public.py`，旧的远程脚本不代表本发行版入口。历史结果不自动等于当前版本的模型成绩。
- 不含密钥、供应商个人账本或第三方书页；因缺失可再分发图解的外部题不进入本版集合。

验证边界与本次检查结果见 [docs/PUBLIC_RELEASE_VALIDATION.md](docs/PUBLIC_RELEASE_VALIDATION.md)。
