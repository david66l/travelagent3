# TravelAgent2 项目已有数据盘点

> 后续全量云端审核已完成：205 个原生动作可复用；六批旧 6,624 条不能原样训练，提取 3,612 个待审迁移请求。以下是前期 manifest 盘点，最终可用性以 [审核报告](EXISTING_DATA_AUDIT_REPORT_20260905.md) 为准。

2026-09-05；用户澄清查询对象是项目已有资产，而非外部通用数据。此次读取本地小型 manifest 与云端正式导出 manifest，没有本地训练或重型数据处理，也没有模型 API 调用。

## 结论

项目有合适的数据基础，也有多批千条级历史资产。当前优先事项应是盘活项目已有 train 场景和轨迹，再决定外部补充；不能只根据最近 24 条试验得出“项目没有数据”。但历史来源存在版本重叠及旧控制器协议，不能把所有目录的行数相加当作当前模型可直接使用的数据。

## 当前主线资产（云端 manifest 实查）

| 资产 | 已有内容 | 用法 |
|---|---|---|
| harness-sft-first-20260905/sft-data-v1 | train 145 动作；validation 38 动作；32 条教师轨迹合计导出 183 动作 | 24 个 train 任务的首轮 SFT 数据；当时已审核并实际训练。来自 v6，继续用于 v9 时应重新核对当前终止/证据语义 |
| harness-student-corrections-20260905/correction-sft-v1 | 12 条教师续跑、train 60 动作、1,346 个监督 token | v9 真实学生状态纠错，已完成来源、独立执行结果和真实 TRL 标签审计；下一轮纠错 SFT 的直接候选 |
| harness-student-corrections-20260905/student-train-v9 | 24 个任务、20 个通过、168 个已验证执行后状态 | 学生错误分布、教师续跑起点；不能直接把学生全部动作当正确标签 |

前两项共有 205 个已导出的 train 动作条目，不是 205 条完整任务，也不是完成了 v9 合并复审的新数据集。validation 的 38 动作及 locked test 不加入训练。v8/v9 的重复评测及教师复采不能视为新独立任务。

## 项目历史资产（读取现有 manifest，未做本轮内容审计）

根目录：E:/A_Louis/TravelAgent2/ml/agentic/datasets。

| 目录 | manifest 中 train 数 | 来源及判断 |
|---|---:|---|
| qwen3-stage20-teacher-formal-3804-v1/sft | 932 动作 | 名称与 candidate_episodes 为 3,804，实际导出总动作 1,362；policy 标记 ControllerFirstPolicy:rollout，不能称为 3,804 条当前模型自主示范 |
| qwen3-stage20-teacher-sft-aligned-v4 | 2,144 动作 | 包含 ControllerFirstPolicy 与旧 verifier-chosen 策略，涉及多个旧专项环境；场景/错误类型复用候选 |
| qwen3-stage32-student-sft-replay-v1 | 1,264 动作 | 源自 stage28/stage32 混合和旧环境，可能与其他版本重叠；先做来源去重和协议检查 |
| stage3-decision-loop-sft-v3 | 1,024 动作 | synthetic，policy 包含 ControllerFirstPolicy / CurriculumTeacherPolicy；不能按名称当作当前自主 loop 数据 |
| native-react-verifier-repair-sft-semantic-v2 | 1,080 动作 | synthetic，react-verifier-repair-corpus.v1；错误恢复题材相关，适合优先审查场景迁移，不直接认可旧标签 |
| native-react-verifier-tradeoff-grounding-sft-v1 | 180 动作 | synthetic，react-verifier-repair-corpus.v3；约束权衡题材相关，需按当前用户授权和终止契约重新执行 |

以上数字是现有清单披露值，不是本轮逐行验证值，也不相加作为独立规模。各目录已有 validation/test，必须保留其来源归属；同源训练与评测派生都不能重新随机混洗。历史偏好、GRPO 及 holdout 目录亦存在，本次没有读取隐藏测试内容或声称其仍能直接训练。

## 推荐的复用顺序

1. 原生数据优先：复核首轮 145 动作在 v9 下的语义，保留已审核的 60 个纠错动作及其完整来源。
2. 从旧 train 的 verifier-repair、tradeoff、aligned 数据开始审查任务请求、可还原工具事实和用户约束。保留原 source_group、split 与派生关系。
3. 分类处理：能还原的场景在当前单一 Agent Loop 重跑，得到新的真实轨迹；只剩合成单步答案的样本仅作为错误类型参考；验证/测试与其同源派生保持隔离。
4. 重新采集时区分学生自主成功、教师全程示范和学生状态教师续跑。旧控制器自动执行的步骤不能通过改名伪装成模型动作。
5. 云端先抽查 100 个旧 train 源任务，统计可还原、重复、协议不兼容和需要补事实的比例，再估算可回收独立任务数。教师使用 GLM-5.3-Flash，每批 API 总费用不超过 100 元，审查与重试计入预算。

下一步应先完成已有资产的云端兼容性抽查，而不是立即采购/合并外部通用数据。外部候选审查报告保留为备选；本次没有启动新的模型采集或训练。
