# Thesis narrative decision — 10 September 2026

Status: user-approved direction for subsequent thesis planning, drafting and revision.

## 1. One central question, three connected steps

中心问题：当公司提供的 aspect taxonomy 增加了训练监督中未出现的 aspect 时，模型如何识别新 aspect 并判断其 sentiment，如何利用不同模型的阶段优势改进分类，以及这种改进在多大程度上依赖训练数据的构造方式？

正文统一遵循：

**找到瓶颈 → 利用阶段互补改进 → 检查改进对训练数据构造的依赖。**

1. **找到瓶颈。** 在 supplied-aspect taxonomy generalisation 下，将错误拆分为 aspect detection 与 conditional sentiment prediction。模型比较、matched one-stage/two-stage controls 和 stage diagnostics 服务于这个问题，而不是独立的排行榜。
2. **利用阶段互补改进。** 解释为什么将 few-shot Qwen 的 aspect-detection stage 与 QLoRA 的 sentiment stage 固定组合。保留 validation selection 到 locked official evaluation 的真实顺序，报告平均提升及影响解释的重要异质性。它不是 learned router 或重新设计的 MoE。
3. **检查训练数据构造的依赖。** 比较删除包含 held-out aspect 的完整 training reviews，与保留 reviews、屏蔽 held-out-aspect supervision。问题是前两步的结论是否依赖这一构造选择，不是证明哪位导师正确，也不是把较高 F1 当作协议更合理的证据。

候选 aspect 仍由外部提供。本文不声称能够发现完全未提供的 aspect。

## 2. Main-text priorities and subtraction

| 内容 | 正文职责 | 减法与完整保留位置 |
| --- | --- | --- |
| 原始模型 benchmark | 一次紧凑比较，交代瓶颈与研究背景 | 完整 per-fold、多指标和阈值细节放 appendix，不重复建立排行榜 |
| Matched one-stage/two-stage controls | 支撑任务分解的设计依据 | 保留 matched scope，不暗示已对所有模型或两种训练策略完成同类验证 |
| Few-shot、QLoRA 与 fixed composition | 解释阶段互补、选择过程和锁定评估 | 合并重复的 stage diagnosis 与错误解释，不重复列出相同数字 |
| 两种训练策略 | 检查核心结论的适用范围 | 正文重点关注 few-shot、QLoRA 和各自策略下的 fixed composition。其他模型完整报告于 appendix，重要反例或改变主结论的结果仍须在正文交代 |
| Minimal descriptions | 支撑 supplied-aspect interface 的解释 | 保留一个集中的问题与结果，不再发展成独立论文主线 |
| Rich descriptions | 一段有边界的补充发现 | 设计与搜索细节移至 appendix |
| Level 1 | 小表与短段作为 closed-taxonomy reference | 不作为新研究主轴，不与 Level 2 直接作纯泛化损失相减 |
| Level 4 | 紧凑的 structured-shift stress evidence | 三组各自呈现，详细结果放 appendix，不再重复完整模型分析 |
| Level 3、Router、calibration、retrieval、DCWT 细节 | 只保留理解主线必要的短说明 | 历史、负结果与完整细节放 appendix。DCWT 仍保留在完整模型 roster 中 |

新证据应替换重复解释、合并同类诊断，不只是新增章节、表格和 caveats。移至 appendix 不等于删除数据。不以结果有利与否决定是否保留。

## 3. Evidence and comparison boundaries

- 原始 review-filtered validation、模型选择和 official-test 结果保持原样。补充研究是在原始 test 后提出的 validation-only comparison，不能追溯改写为原始 test 前的计划。
- 正确 masking 不把 held-out aspect 标成负类。它保留 review text，但不生成该 aspect 的正负训练问题，也不提供其 sentiment supervision。
- 两种策略保持模型 family、folds、validation rows、candidate resources、训练预算、decoder 和 selection rules 的配对定义。相同选择规则不保证选出相同 learning rate、checkpoint 或 threshold。任何额外差异均须记录。
- 截至本决定，已完成的补充研究涵盖 TF-IDF、DCWT、DistilBERT、few-shot Qwen，以及 E5/zero-shot Qwen 的不变性检查。现有混合诊断仍使用旧 review-filtered QLoRA sentiment component，不能称为完整 masking-trained hybrid。
- 尚需补齐 masking QLoRA、严格可比的 filtering reference、N/D scoring、对应策略下的完整 composition 及其审计，才能回答完整两阶段组合的训练策略依赖问题。不得提前写出结果。
- 若补充后组合优势减弱或消失，应据实收紧结论。原始 locked-test finding 不被抹去，新策略也不被包装成已得到 official-test confirmation。
- 不重新搜索 router、prompt、模型组合或最佳指标来维持优势。任何新的 official-test evaluation 必须另行明确规划与授权。

## 4. Writing and review standard

- 读者是机器学习专家，但不熟悉本项目。解释任务特有术语和具体操作，不重复普及通用机器学习知识。
- 每节只承担一个明确问题。Results 说明发现，Discussion 解释原因与意义，不逐表复述。
- 采用 Griffin 批注所强调的具体、清晰写法，避免压缩过度的术语堆叠和分号。
- 必要限制集中在相关论证处，不在摘要、结果、讨论和结论反复叠加同一防御性表述。
- 保留重要的不利发现和敏感性分析。只删除实质重复，不为页数强行压缩已审核文字或版式。
- 后续改稿使用可交互的逐项审阅，记录改前、改后、理由、证据来源，以及移至 appendix 的位置。保留旧版本和旧审阅决定。

## 5. Authority and current action boundary

本文件优先于旧写作计划中与以上叙事、内容优先级冲突的建议，但不替代实验 manifest、结果文件或原始 test governance。

本次授权仅为记录方向和起草给 Dr. Aji Ghose 的 Slack update。尚未授权在本轮开始新 draft、QLoRA 训练、GPU 计费、test 或 GitHub 提交。消息为用户发送的草稿，不由代理直接发出。

Evidence entry points:

- the author-only chronological research notes (not distributed publicly)
- `experiments/taxonomy_training_policy_extension_v1_results.md`
- `experiments/taxonomy_training_policy_sensitivity_v1_results.md`
- `../../Thesis/Training_Policy_Study_20260908/`
- `../../Thesis/Griffin_Revision_20260908/`

The existing research report is an evidence source, not an instruction to preserve filtering because its F1 is higher. Interpretations must follow the boundaries above.
