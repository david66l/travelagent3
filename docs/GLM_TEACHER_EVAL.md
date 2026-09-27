# GLM 教师评测怎么运行

这里接入的是教师 API。它通过现有 `NativeToolAgentPolicy`、`SelfRepairingAgentPolicy` 和 `BoundedAgentLoop` 选择动作，没有第二套决策流程。

## 配置和运行

密钥放在项目根目录被 Git 忽略的 `.env` 中，变量名 `ZAI_API_KEY`。国内通用 API 地址为 `https://open.bigmodel.cn/api/paas/v4`，模型为 `glm-5.3-flash`。不要将 `.env` 放入结果压缩包。

将 `ZAI_API_KEY` 载入进程环境后，从项目根目录运行：

```bash
python scripts/evaluate_harness_teacher.py \
  --output /absolute/path/to/new-run \
  --model glm-5.3-flash \
  --base-url https://open.bigmodel.cn/api/paas/v4 \
  --concurrency 4
```

本地使用项目 `backend/.venv` 的 Python；云端使用已安装求解器和评测依赖的环境。`--case query-01` 可以限定某一题，重复 `--case` 可以限定多题。

脚本直接读取进程环境，不会自动读取 `.env`。支持 python-dotenv 的环境也可以用 `python -m dotenv -f .env run -- python scripts/evaluate_harness_teacher.py ...` 载入配置；`--base-url` 明确指定平台，脚本不会根据密钥猜测平台。

## 本轮固定了什么

- 使用 `evaluate_harness_base.py` 原有 24 个开发诊断案例及同一判分函数。
- 使用固定研究数据、实际路径矩阵、实际求解器、实际校验器；没有实时查询互联网或调用地图服务。
- 意图和约束已预先声明，因此本轮不评测意图解析和用户回复后的多轮恢复。
- 每题最多 24 步、24 次工具调用、3 次求解、32,000 个输出 token；思考 token 计入输出。
- GLM 每次请求最多输出 8,192 token，启用 max 思考，每题最多 600 秒、单次请求最多 180 秒。Qwen 原基线为不思考、贪心解码、每次 384 token、每题 120 秒，因此不是等算力比较。
- GLM 接口使用 `tool_choice=auto`。一次返回零个或多个工具调用都判为失败，现有自我纠错机制最多再尝试一次。不会截取第一个调用冒充成功。
- 在原有 system 和状态消息后补一条“每轮只调用一个工具”的协议提醒，不补工具顺序或答案。状态内容与原有投影相同。每轮根据当前状态重新请求，不跨轮携带私有思考文本。

## 结果文件怎么看

| 文件 | 含义 |
|---|---|
| `manifest.json` | 模型、接口、预算、解码参数、代码哈希和运行环境 |
| `cases.json` | 全部诊断题 |
| `case-id/model-calls.jsonl` | 每次请求的状态、工具协议、原始响应、动作、错误和用量；不含认证请求头 |
| `case-id/episode.json` | 同一 Loop 实际执行的完整轨迹 |
| `case-id/tool-calls.json` | 工具的真实参数和返回结果 |
| `case-id/summary.json` | 单题判分、耗时、用量和故障分类 |
| `summary.json` | 本次已完成题目的汇总；24 题整体结论要求全部落盘，定向复测单独报告实际样本量 |

`reasoning_content` 仅作为私有审计证据，不直接作为学生的训练目标。API 若未提供思考 token 明细，记录未知及已知调用数，不凭文字长度估算。题目到时限取消的请求也可能已经计费，但客户端没有完整用量；报告中的已知 token 不等于账户最终账单。

## 防止重跑污染

同一目录仅允许配置和源代码完全一致的续跑。完整题目跳过；已有请求日志却缺少单题汇总的中断题目拒绝自动重跑，使用新目录保存新的实验。鉴权、限流、服务故障会停止排队题目；单题达到自身时限记为该题失败，不当作平台故障。

检查结果时，应重新验证轨迹哈希，并从真实求解输入和输出独立校验最终行程。不能把合法 JSON、流畅解释或“已经求解过”当作任务成功。

## 这批诊断数据不能直接拿去训练

它曾用于发现和修复系统问题，属于开发数据。教师跑通之后，应另建任务，按场景模板、实体和来源与 dev/test 分组隔离。

终止契约已升级至 `terminal-evidence-v4`：当前详情中的明确闭馆日期或必付费用下界可以生成证据 ID，模型自主选择并引用证据停止；检索遗漏和单次求解失败不代表任务无解。校验器版本为 `travel-validator.v2`，判分版本为 `terminal-contract-v3`。最新 8 题定向复测见 `TERMINAL_CONTRACT_FIX_20260905.md`，不替换历史 24 题原始结果。

闭馆诊断的搜索正文已与结构化详情对齐，fixture 版本为 `closure-consistency-v2`。旧、新两轮之间同时有此数据变化，不能把差值都归因于 Harness。餐厅查询仍只得到景点或泛化结果，尚未修复；正式生成示范前必须单独修正并测试。

官方接口说明：[智谱对话补全](https://docs.bigmodel.cn/api-reference/模型-api/对话补全)。实际返回的模型标识、字段和用量以本轮原始响应为准。


## 新训练任务入口（2026-09-05）

现可用 `--cases-file ML/agentic/data/harness_training_pilot24_v1/cases.json` 运行新 train 任务；仍复用同一 Loop。最新执行修订为 `terminal-rejection-recovery-v5`、fixture 为 `research-database-v4`。6 题抽样与资料边界见 `RESEARCH_DATA_PILOT_20260905.md`，它不替换历史诊断结果。
