# 情感阶段服务一致性：保留完整参考批次

日期：2026-11-11。状态：实现、CPU 检查、4条×2方法真实 GPU 核验、100条×2方法 API 核验及真实模型故障检查均完成。

## 发现与证据范围

在最终 HTTP 服务测试开始前，使用同一批 100 条开发集输入，将接口输出与冻结的离线分数、既定阈值进行逐条比较。这是模型输出一致性检查，不读取人工金标，也不使用 957 条评估集调整方案。

- 固定示例方法的最终标签集合为 **100/100 一致**。
- `e5_negative_seen` 方法的最终标签集合为 **99/100 一致**，全部请求成功。
- 后者的方面存在性概率和原始 logit margin 最大绝对差均为 **0**，检索提示合同不匹配数为 **0**。
- 后者情感概率的全批最大绝对差为 **0.013985216617584229**。唯一标签集合不一致的评论中，总体满意方面的中性分数从离线的 **0.3999535143375397** 变为接口的 **0.40524381399154663**，跨过原有的 **0.402159124613** runner-up 阈值，接口因此多输出了一个中性标签。

原始失败证据完整保存在本地 `outputs/rag_20261111/service_final/`，不覆盖、不将失败版本作为通过的性能测试。本文不记录评论文本或私有行标识。

## 原因定位与验证

原离线情感参考对全部 12 个候选方面评分。当前配置为每微批最多 4 条提示、最多 4096 个填充后 token，QLoRA 提示最长 384 token，因此单条评论的完整参考始终形成三个保序的四条微批。

旧接口先筛出被检出的方面，再重新将这些情感提示组成微批。上述边界案例只执行了两条情感提示，填充形状为 `2 × 132`；其提示正文没有改变，但同批候选、填充形状以及最后 token 输出位置集合可能与完整参考不同。在量化模型中，这些执行形状变化可能带来数值差异，进而影响临界阈值判断。

后续真实 GPU 小样本验证中，保留原分组后实际输入、分数及标签均与完整参考一致，支持执行形状导致边界差异的解释。随后两种方法均通过全部100条 API 一致性检查，结果见文末已完成门槛。

## 修正：按完整参考组条件执行

新增运行模式 `sentiment_execution="canonical_full_grid_groups"`，流程如下：

1. 按原有候选顺序编码完整的 12 条情感提示，调用原微批规划器。
2. 保留至少包含一个已检出方面的完整微批；其余微批整体跳过。
3. 被保留的微批使用原来的全部候选作为同批输入，保持输入顺序、填充形状及最后 token 位置映射。
4. 同批辅助候选参与前向计算，但接口仍只输出被检出的方面。

固定示例和 `e5_negative_seen` 两种方法在新服务版本中都显式使用该模式，避免只给某一种方法更有利的执行条件。原 `selected_only` 模式保留用于历史复现；新模式不改变完整参考路径和公共情感分数缓存的模型输入。

**模型、适配器、方面存在性分数、情感解码规则、阈值、示例选择策略及已冻结的方法选型均不改变。** 这是接口与既有离线分类器之间的执行一致性修正，不增加一轮算法调参，也不通过移动阈值掩盖分数差异。

## 计算成本与计数口径

该模式仍需在 CPU 上编码所有情感提示。GPU 只执行命中已选方面的完整组：在当前单请求、12 候选、每组 4 条的配置下，实际执行提示数只能是 0、4、8 或 12。

相对于旧的“仅把已选方面重新组批”方式，同批辅助候选会增加计算量；相对于完整 12 条评分，可以跳过没有已选方面的整组。若三个组均被命中，则不节省情感前向计算。实际端到端延迟、吞吐和显存表现须重新测量，不从理论组数直接推算性能收益。

响应中的 `selected_aspect_count` 单独记录输出方面数；`sentiment_prompt_count` 记录实际前向提示数，包含同批辅助候选；`sentiment_companion_prompt_count` 记录这些辅助候选数；`sentiment_full_grid_prompt_count` 记录完整参考候选数。这些字段按每条请求统计，`batch_sentiment` 内的字段按整个推理批次统计。

`sentiment_group_contract` 绑定执行模式、候选有序指纹、每批提示数、填充 token 上限和 logits 模式；模型版本带有该合同的指纹。实际执行组的位置另行记录。一般配置只保证与同一请求批次的完整参考一致；当前服务每次请求批次为 1，且 12 个候选恰好分为三个四条组。

## 已完成检查

已完成的 CPU 检查包含一个显式依赖批次形状的模型：它能够复现旧条件执行的分数漂移，并验证新模式保留完整参考组后，所选向量与标签严格一致。测试还覆盖空选择、只命中一个组、命中不连续组、全部命中、token 预算拆组、跨请求计数及组身份变化。相关核心检查为 **68 项通过**；新增模块的离线运行身份绑定检查为 **5 项通过**。

真实 GPU 检查已完成，使用已发现的边界输入和三条按文本长度选择的开发输入，分别对固定示例和 `e5_negative_seen` 验证以下三项，全部通过：

- 完整参考与新条件执行的实际填充输入、注意力掩码及最后 token 位置映射完全一致；
- 所选方面的情感向量与最终标签完全一致，并与原冻结公共情感分数缓存一致；
- 方面存在性分数及检索策略保持一致。

小规模 GPU 门槛通过后，已在新输出目录 `service_final_v2` 完成两种方法各100条 HTTP 一致性检查及真实模型故障检查，均通过。服务性能测量排在冻结的历史质量回放后执行；一致性通过不替代性能实测。旧失败目录完整保留。

核验入口：`scripts/check_canonical_sentiment_gpu.py`；执行策略与合同：`src/msc_project/serving/sentiment_execution.py`；实现与测试：`src/msc_project/serving/inference.py`、`tests/test_canonical_sentiment_groups.py`。
# Native GPU evidence

The fresh canonical gate completed on 11 November 2026: four development inputs
per method, including the original boundary case and three input-length strata.
Both fixed and `e5_negative_seen` passed. Each executed sentiment microbatch had
identical collated input IDs, attention masks and logits positions to its original
full-grid group. Selected sentiment vectors and final labels were exactly equal
to the full-grid reference and the pinned historical sentiment anchor (maximum
selected probability difference zero). This is an input/score consistency check,
not classification accuracy. No service performance result follows from this gate.

Private evidence: `outputs/rag_20261111/canonical_sentiment_native_gate/summary.json`.
Service plan: `service_final_v2`, SHA256
`c7e4069ea2a72983ad4dfd8f6bde5dd4bcdade1eb616d336b0b5b85113f60a57`.

## Completed API and fault gates

Both methods completed all100 requests successfully with100/100 exact final
label-set matches and no retrieval/prompt contract mismatch. Dynamic presence
probabilities, raw margins and selected sentiment probabilities were exactly
equal to the frozen offline reference. Fixed selected sentiment probabilities
were also exact. Actual fixed sentiment prompts were592, including403 companion
prompts; dynamic prompts were620, including390 companions. These are executed
Stage2 prompts, not returned aspects.

Fixed prefix caching changed presence probability values (maximum absolute
difference0.12068390846252441) and raw margins (maximum1.0625), despite no changed
final labels in this100-input gate. Therefore a separately registered fixed
cache-on HTTP957-review deployment-quality replay will follow the frozen offline
matrix. It cannot change classifier selection or thresholds. Old serving957
results use a different prompt/scoring/execution path and are not a substitute.

The real-model fault gate completed13 execution steps,6 paired comparisons and4
pre-scoring rejection checks, all passing. It covered normal timed retrieval,
exception/timeout/busy fallback, late-result recovery, altered legal index
identity and restoration, with frozen fixed fallback thresholds. Evidence is
under `service_final_v2/correctness/` and `service_final_v2/fault_gate/`.

The957-review eight-method historical inference matrix started after these gates.
