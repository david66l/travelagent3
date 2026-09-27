# TravelAgent2 公开数据复用审查

核查日期：2026-09-05。状态：官方资料与公开 schema 初审完成；未导入训练、未运行外部基准、未支付模型 API 费用。

## 结论

存在值得复用的数据，应采用“公开通用工具训练数据 + 旅行任务/沙盒 + 本项目真实学生纠错”组合。无需从零用教师生成所有内容，但公开行数不能直接等同于本项目已验证的完整旅行任务数。

优先审查 When2Call 与 ToolACE 的通用辅助训练接入；Open-Travel 是中文旅行任务扩充的重要研究候选；ChinaTravel 和 TravelBench 优先作为外部验证环境。TravelPlanner 官方训练示范仅 45 条，不能把其 1,225 条总量误当训练规模。xLAM 60k 是备用，当前访问需要平台条件确认，不从第三方副本绕过。

## 候选核对

| 数据 | 官方披露规模/形式 | 数据许可 | 本项目用途与限制 |
|---|---|---|---|
| [NVIDIA When2Call](https://huggingface.co/datasets/nvidia/When2Call) | SFT 15,000；偏好 9,000；另有测试 | CC BY 4.0 | 调用、追问、无法回答的决策边界很相关。优先抽样；不是完整旅行恢复轨迹。偏好与 SFT 可能同源，不能当完全独立任务计数 |
| [Team-ACE ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | 公开 train 约 11.3k 对话；system + conversations | Apache 2.0 | 多样工具选择与多轮对话的辅助训练。工具名称、参数及部分调用表示与本项目不同；先检查可解析性和语义，不能直接改名映射 |
| [Alibaba Open-Travel](https://huggingface.co/datasets/Alibaba-NLP/Open-Travel) | 中文 RL train 1,626；test 250 | CC BY-NC 4.0 | 领域任务来源优先候选。训练文件提供请求，不能当作已有合格工具调用示范；需评估环境支持并重新执行，研究资产单独管理 |
| [ChinaTravel](https://huggingface.co/datasets/LAMDA-NeSy/ChinaTravel) | Easy 300、Medium 150、Human 154；另有 Human1000 与 Phase 2 等配置 | CC BY-NC-SA 4.0 | 中文多约束、结构化行程、可执行判据及配套沙盒适合外部评测。不是现成 agent SFT 轨迹；默认全部按评测候选隔离，不自行把 easy/medium 当 train |
| [TravelBench](https://github.com/small-xiangcheng/TravelBench) | 单轮、多轮、不可解任务；10 类旅行工具及缓存环境 | 数据 CC BY-NC 4.0；代码 MIT | 多轮与能力边界很相关。默认依赖其他模型作用户/工具模拟与判分，以及嵌入服务；不直接运行其默认收费脚本 |
| [TravelPlanner](https://huggingface.co/datasets/osunlp/TravelPlanner) | train 45、validation 180、test 1,000 | 数据 CC BY 4.0；代码仓 MIT | 旅行约束和外部基准参考。英语/美国旅行、包含住宿航班等，和当前国内工具/求解字段不同；45 条计划也不等于 45 条动作轨迹 |
| [Salesforce xLAM 60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | 60,000 条函数调用样本 | CC BY 4.0；当前页面要求登录确认访问条件 | 通用辅助训练备用。不能把函数调用总量当完整旅行任务数；不绕过访问条件，不是当前急需的过期证据恢复数据 |

许可来自各自数据卡，不能用代码仓 MIT 代替数据许可。NC/SA 数据及派生的实验资产保留独立来源与许可记录；商业部署或再分发前需按实际用途确认授权，不能仅凭“开源”认定可任意使用。

## 与当前架构的兼容性

当前学生输入是 PolicyContext，输出一个动作及语义参数；可信实体、约束和工具执行参数由 Harness 装配，只有十三个线上动作。公开通用工具数据通常包含自己的 tools schema、对话消息和不同调用格式。两者的差异是真实训练分布差异，不能用字符串替换解决。

- 通用辅助训练保留来源工具规范，使用独立的数据类型和模板转换；它不意味着线上增加任意工具权限，也不能计入“已在 TravelAgent2 执行通过”的轨迹数量。
- 旅行请求可作为任务种子，先核实我们的工具与求解器支持对应城市、时间、住宿/交通/餐饮条件，再采集实际完整轨迹。无法表达的约束明确拒绝或记录待实现，不能静默删除。
- Open-Travel 的 1,626 条是 RL 请求，并未提供当前 Agent 的逐步正确标签。其价值在于真实类型任务来源，不是省掉所有教师执行。
- When2Call 已有同输入偏好字段，值得独立研究 DPO；接入本项目时仍需保留真实输入、工具定义及偏好语义，不把自由文本追问直接当已执行 ask_user。
- 中文/英语、单轮/多轮、并行/串行调用分别统计；多个动作一并输出的样本不能无条件拆成多个具有虚构中间状态的训练步骤。
- 学生已会生成多数合法调用，通用数据不应淹没旅行决策数据。先比较无通用辅助、少量辅助两组，再依据本项目 dev 调整监督 token 配比，不预设固定混合比例必然有效。

## 测试隔离与可信评测

公开 benchmark 不作为当前项目自己的秘密 test。公开测试的存在不代表基础模型从未见过，报告需要区分公开外部评测与项目保留测试。

ChinaTravel 的 Phase 2 full 2,000 条中包含正式比赛使用的 100 条，官方字段 official_phase2_evaluation 标记该重叠；不能把 full 全部拿去训练后再报告该 100 条测试成绩。When2Call 的 300 条 LLM judge 测试是 3,652 条 MCQ 测试的子集，两者不能相加计独立测试数。各源配置与派生关系在导入前登记。

第三方判据（如 ChinaTravel 的 hard_logic_py）属于隐藏评测资产，不能放入模型提示，也不未经审查在主服务执行任意 Python。外部沙盒若使用独立工具集合，应明确称为外部环境实验；必须同时保留当前 Harness 的原生评测，不能用外部高分替代上线链路验收。

TravelBench 官方配置列出 GPT 用户/工具模拟、Gemini 判分与 Qwen 嵌入依赖。用户已指定 GLM-5.3-Flash 和 API 预算，不能直接照跑这些配置。替换为 GLM 或固定用户脚本后需重新校准，结果应标注修改环境，不能直接冒充官方同协议成绩。

## 费用与推荐顺序

教师指定 GLM-5.3-Flash；每批 API 总开销不超过 100 元，包含审查、失败和重试，云端算力另计。本次资料审查没有调用任何模型 API。正式采集前核实实际端点的当前单价、思考 token 计费及缓存口径；本次未核实出可用于预算执行的官方单价，不据第三方价格承诺能生成多少条。

1. 云端取得公开数据的固定版本、许可、数据卡和 train 小样本；先检查 When2Call 与 ToolACE。先审核每源 100–200 条，再决定清洗完整 train；抽样不是正式训练规模。
2. 按来源、工具 schema 和派生关系隔离评测；检查重复、缺参、角色、工具调用 ID、输出格式和实际监督 token。
3. 对 Open-Travel 做环境兼容性分层审查；按研究许可单独管理。只在可执行且费用有界的请求上用 GLM 采集，不批量把问题改写成虚构答案。
4. 保留本项目真正缺少的学生失败纠错：过期证据、实体错配、无进展与版本失效。教师预算优先花在这些公开通用数据无法直接覆盖的状态上。
5. ChinaTravel / TravelBench 分别评估判据与沙盒接入成本，形成外部评测；TravelPlanner 用于补充，xLAM 暂作备用。

原规模计划中的 2,000–5,000 条独立旅行任务目标保留。公开通用数据的数万条对话是另一项资产，不用于填充该领域任务指标。

## 本次访问证据和限制

- 已通过官方 Hugging Face 数据卡、作者代码仓与论文入口核对上述说明；网页预览不是完整数据质量审计。
- 云端 metadata/训练小样本探查失败：Hugging Face 返回网络连接或超时错误，显式 IPv4 探查也超时。未将文件下载到本地绕行。
- 云端 raw.githubusercontent.com 的 TravelPlanner README HEAD 探查返回 200，后续可优先使用官方 GitHub 入口；ChinaTravel 官方数据卡还提供 ModelScope 沙盒镜像，可后续验证。
- 云端网络诊断摘要：/root/autodl-tmp/travelagent-public-data-review-20260905/summary.json。该文件记录访问失败，不是成功获取的数据 manifest。
- 尚未固定数据 revision、完整下载、执行第三方代码、适配训练模板或训练任何模型。所有候选仍需导入前审核。
