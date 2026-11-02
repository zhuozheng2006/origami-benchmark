# 历史题目全量审计 — 2026-11-02

上次 public-v1 只发布刚重新跑过的 O01–O03，范围过窄，遗漏了早期可用题。这次按完整数据目录、冻结清单、参考报告、轨迹及评分器盘点，没有重新爬取或生成新题。

## 结论与口径

- 早期 D/H/J/S 四批 144 道：原冻结 SHA256 全部一致；用现有评分器读取原始轨迹、metrics、事件、动作、检查点重新计算，121 通过、23 不通过。数据和答案完整公开，失败报告也保留。
- O01–O03：上一版独立构建后端的真实重放全部通过，保持在正式多层轨道。
- 七道原站子任务曾通过物理验收。其中五道可独立下载并逐像素还原原图；H03/D05 的原站返回 403，列为素材获取待修复，不进入默认分数。
- Ultra 100：只有 12 道具备完整图解/目标与完成的物理参考轨迹；本次重新评分均通过旧规则，但旋转/高度/路径严格性问题仍存在，全部列为待修复评分的诊断题。其余 88 道没有完整验收链，不能称为已完成 100 题。
- Intermediate advanced 100：100/100 的初始顶点等于目标顶点，存在手写 verified 标记、未支持宏操作和非闭环“评测”。列为无效数据，而不是前沿模型的 0% 成功率。其 3 道前身题也没有可执行目标/受支持完整参考路径，列为无效规范。
- 旧 v1、v14/v15/v16 与修订数据有重复/被替代记录。保留其身份及历史结论，不把不同版本计成独立新题。34 个整件作品题源也不是 34 道可执行 benchmark 题。

最终默认可运行 129 道：124 道仓库自带图解，5 道须原站获取。注册表另含 13 道诊断题与 2 道素材阻塞题，共 144 个可寻址任务 ID。

## 逐数据集版本记录（不是独立作品计数）

| 数据集 | 已验收记录 | 待修复/补齐/旧版本 | 无效 |
|---|---:|---:|---:|
| diagram2fold-generalization-v1 | 8 | 0 | 0 |
| diagram2fold-generalization-v2 | 7 | 1 | 0 |
| diagram2fold-small-v1 | 99 | 21 | 0 |
| diagram2fold-tiered-v1 | 0 | 34 | 0 |
| diagram2fold-v1 | 0 | 8 | 0 |
| diagram2fold-v1.1 | 7 | 1 | 0 |
| document-hard-v13 | 2 | 3 | 0 |
| document-hard-v13-amendment | 0 | 1 | 0 |
| document-layers-v14 | 3 | 3 | 0 |
| document-layers-v14-real-extension | 1 | 0 | 0 |
| document-layers-v15 | 3 | 2 | 0 |
| document-layers-v16 | 4 | 2 | 0 |
| document-layers-v17 | 4 | 4 | 0 |
| document-pilot-v12 | 3 | 6 | 0 |
| intermediate-advanced-100 | 0 | 0 | 100 |
| intermediate-advanced-v1 | 0 | 0 | 3 |
| ultra-hard-100 | 0 | 100 | 0 |

所有 430 条记录见 [inventory.json](../audit/inventory.json) 和 [交互表](../audit/inventory.html)。三个分类指当前数据/验证状态；“无效”不表示真实折纸在物理上不可能。

## 保留证据

- [逐步图解的重新评分、原报告与输入哈希](../audit/legacy-reference-audit.json)
- [复杂多层参考的成功及失败重新评分](../audit/ultra-reference-audit.json)
- [参考报告与失败事件](../audit/reference-reports/)
- [公开运行注册表](../data/evaluation-catalog.json)

历史参考通过证明在所标注协议下存在有效路径，不保证任意操作、硬件或第三方模型都成功。本次公开适配只改运行地点和发布接口，没有按题写求解分支；逐步/整页/多层协议不合并计分。
