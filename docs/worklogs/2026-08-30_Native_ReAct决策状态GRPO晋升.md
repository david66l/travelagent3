# 2026-08-30 Native ReAct 决策状态 GRPO 晋升日志

## 本阶段目标

让 RL 在真实 TravelAgent 决策上产生可复现的指标提升，同时保证完整 Agent Loop 不退化。训练、专项评测和全链路评测均使用生产 ReAct 工具协议，不使用静态“猜答案”替代真实状态转移。

## 主要问题与定位

1. 早期整轨迹 GRPO 数据与生产动态工具 schema 不一致，出现全负奖励或零方差，无法形成有效梯度。
2. SFT 在真实工具历史后的 `get_poi_detail` 决策状态存在参数幻觉；模型会复制 `candidate_poi_ids`，但这些可信身份应由控制器从 ledger 注入。
3. 原始 GRPO 候选虽然在专项集从 42/48 提升到 43/48，但直接替换全局策略后曾降至 17/32，说明窄数据更新造成行为漂移，不能晋升为通用模型。

## 完成的工程改造

- 从真实审计轨迹抽取 decision-state replay 数据，保留已验证工具历史，并严格隔离 train/validation。
- 使用真实工具调用、状态转移和程序化 Verifier 作为 GRPO reward；无正负样本方差的组不形成有效更新。
- GRPO 参数收敛为：group size 8、learning rate 5e-7、KL beta 0.01、1 个受控优化步。
- 在策略权限边界丢弃 `get_poi_detail` 中模型重复提交的 controller-owned POI 字段；其他未知业务字段仍硬拒绝。
- 新增决策专家路由：SFT 负责通用 ReAct，只有存在 `poi_candidate_set` 且尚无 `poi_detail_set` 时调用 GRPO；异常自动回退 SFT。
- 新增 vLLM multi-LoRA 启动脚本，在同一 Qwen3-1.7B 基座上挂载两个 adapter，避免常驻两份基座模型。

## 冻结评测结果

| 评测 | SFT | GRPO + KL | 结果 |
|---|---:|---:|---|
| 决策状态冻结集（重复采样级） | 42/48（87.50%） | 48/48（100%） | +12.50pp，诊断指标 |
| 完整 Agent Loop | 88/92（95.65%） | 88/92（95.65%） | 0 回归 |
| 完整 Loop 参数/格式错误 | 3/92 | 3/92 | 持平 |
| 完整 Loop 平均延迟 | 5290.2ms | 5268.8ms | 基本持平 |

专项集实际只有 6 个独立任务，每题重复采样 8 次。按任务聚合后为 4 题改善、0 题退化、2 题持平，双侧精确符号检验 p=0.125，未达到 0.05 显著性标准。原采样级 6 个改善、0 个退化和 McNemar p=0.03125 只能描述同题随机采样稳定性，不能证明跨任务泛化。完整 Loop 的 23 个任务、92 次重复采样结果完全一致。因此该证据只支持把 GRPO 适配器作为 POI 详情专家的 Shadow 候选继续验证，不支持生产晋升或全面替换 SFT。

## 被否决的实验

- `beta=0`、lr=1e-6 的 GRPO：专项仅 43/48，且一轮全链路曾出现 17/32，拒绝全局晋升。
- 第二个无 KL 更新步：专项回落至 42/48，拒绝。
- KL 候选的一个训练种子产生 8/8 全负奖励、reward std=0；该 checkpoint 无有效 GRPO 梯度，拒绝进入评测。

## 产物

- 通用 SFT：`qwen3-1.7b-native-react-sft-decision-bridge-step3-v1`
- GRPO 专家 Shadow 候选：`qwen3-1.7b-native-react-grpo-decision-kl001-lr5e7-step1-seed06-v3`
- 试点审计报告：`artifacts/native-react-posttraining/native-react-rl-promotion-v1/`
- 多 LoRA 服务脚本：`scripts/serve_native_react_multi_lora.sh`
- 部署配置：`deploy/native-react-decision-specialist.env`

模型、训练数据及原始评测报告已从云端校验哈希后保存到本地 `artifacts/native-react-posttraining/cloud-backup/`。传输用临时压缩包在完成校验和解压后已从本地与云端删除。

代码验证：后端全量测试 `1392 passed, 3 skipped`，本阶段修改文件 Ruff 检查通过。

真实服务烟测：云端 vLLM 0.8.5 在同一基座成功加载 `travel-sft` 与 `travel-grpo-poi`；生产策略类通过 OpenAI-compatible 接口验证，专家状态实际命中 GRPO 并调用 `get_poi_detail({})`，普通状态实际命中 SFT 并调用 `get_weather({})`。临时服务在烟测后已停止并释放显存。

## 后续

1. 扩大决策状态任务数、城市和故障类型；当前专项集只有 6 个任务、每题 8 次采样。
2. 将专家路由的命中率、回退率、硬通过率和单决策延迟接入监控，再决定是否扩展到其他动作。
