# H006 checkpoint-32 完整 Agent Loop 评测运行记录

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent run mode
- Mode: run
- Date: 2026-09-04
- Verification Status: INVALID_INFRASTRUCTURE / RECOVERY POLICY OUTPUT ERRORS
- Version Label: h006-full-agent-loop-v1-20260904
- Candidate: H006 checkpoint-32 only
- Invalid preserved attempt: h006-full-agent-loop-v1-20260904-r1
- Executed Run ID: h006-full-agent-loop-v1-20260904-r2

## 白话目的

前一轮只证明模型在单个决策状态下会选动作。这一轮让同一个 checkpoint-32 从真实用户请求
开始，连续完成需求理解、多轮工具调用、证据收集、CP-SAT 求解、Verifier 校验、成稿或安全
停止，并在另一套考卷里注入一次工具故障，检查它是否会立刻正确恢复。

不重跑 H001，不运行 H006 step-34，不打开 promotion-val 或 sealed-160。

## 冻结材料

- adapter：`/root/autodl-tmp/TravelAgent2-h005-eval-20260903/artifacts/native-react-posttraining/h006-internal-corrective-sft-v2-seed20260930/checkpoint-32`
- adapter SHA256：`7a26396eb6d8a4ce477431d78be39d1362bb7173bd29685634936e02fddc25cd`
- base model：`/root/autodl-tmp/models/Qwen3-1.7B`
- chat template：`backend/src/agentic/templates/qwen3_agent_prefix_preserving_v1.jinja`
- chat template SHA256：`dfe4e379b6439a9f01e881660c5f1ea57cd8026a8831553492723b38d15c9e63`
- full-loop benchmark SHA256：`6c49db74b80141979b56a93a5e287f959b1abe93a12eda43c2448fd74e0257d4`
- recovery benchmark SHA256：`cabb73f84624a9f762dbedffe0cec34f22f183131381738f0d3a8de7f15e7037`
- checkpoint-32 完整目录 combined SHA256：`704ef680e4a4d56229f118fabeaf4f7f666da604d4d05cdda926c6545cf1f122`
- base model 完整目录 combined SHA256：`aa81aeaaee547ef68a6a993763cb3ef38a8223eafe2f2eeef139a47f2715110d`
- 云端版本：vLLM `0.8.5.post1`、PyTorch `2.6.0`、Transformers `4.57.6`、PEFT `0.20.0`
- full-loop cases：core 10 + expanded 20
- recovery cases：8 cases × 4 seeded samples = 32 rollouts
- GPU 门禁：必须是 RTX 4080 SUPER、总显存与空闲显存均不少于 30000 MiB
- 端口预检：本机 127.0.0.1:18000 与云端 127.0.0.1:8000 均空闲
- 修正后正式预检测试：98 passed（`-p no:cacheprovider -W error`）

本轮使用 E 盘当前生产代码运行评测；云端只运行模型服务。运行包装器会在实验开始和结束时
分别扫描 `backend/src/**`、`backend/tests/**`、`scripts/**`、`backend/pyproject.toml` 和
`backend/uv.lock`，保存
完整逐文件 SHA256、组合 SHA256、Git HEAD、工作树状态 SHA256、dirty patch SHA256、Python
与依赖版本及非敏感配置。开始和结束快照任一处不同，本轮结果自动判为基础设施无效。

独立审核版本已固化到 `experiments/H006-FULL-AGENT-LOOP-RUNTIME-MANIFEST.json`，共 789 个文件，
并锁定 Run ID `h006-full-agent-loop-v1-20260904-r2`。runtime combined SHA256 为
`50ec0ab50b13b83fb4348adc6505b153f6a8188b15aefbf7ec1edf418dfc66a3`，manifest 文件 SHA256 为
`a374f4029cbac16c1bad63c99935385294e7bf485110bee797d5e7748ef955db`。
正式入口必须收到这个 manifest SHA，并在任何测试、SSH 或模型启动前逐项核对；审核后若代码、
测试、依赖或行为配置发生变化，直接拒绝运行。

修正后的关键文件 SHA256：

- `scripts/run_h006_full_agent_loop_eval.ps1`：`a4b3d0d7d7f253bb00084aad07fb47da5b991c08b1e57e870e487b67a9fc17ac`
- `scripts/start_h006_checkpoint32_server.ps1`：`765c63deb2de8d214af263a6ed314fe013575c7ee719675aacf870c35be49469`
- `scripts/audit_h006_full_agent_loop_run.py`：`354a2f58090816594d2eb3cf807cdc02f8096b182cbb2fe5d8552906ee2329cb`
- `scripts/evaluate_full_agent_loop.py`：`16aecd337ef7c407e4e73816ba501e6654cebfd17ff864982aca464f8d9ccdd7`
- `scripts/evaluate_full_agent_loop_recovery.py`：`c9370d3409e11e6c119872e0923cf3f2770b5672188bd26a65886d2daa3e04e0`
- `backend/src/core/conversation_turn.py`：`cf4b390a224f8892bb248fb0ff2f1035805d3495abf35d98b8ebb6bbb7779587`
- `backend/src/core/langsmith_trace.py`：`b3434f623de6573ecd534d29e61ea07e66153f69abf803820697dc1ba468635d`
- `backend/src/schemas.py`：`7b0cff701d9549adc6bb767df4c78ad464223c1b2fe25707c9e3066c70212573`

## 成功标准

这一轮不把“脚本退出码 0”和“模型通过”混为一谈：

1. 基础设施成功：模型服务健康；每段评测退出码为 0；输出 JSON 非空且记录数完整。
2. 完整链成功：30 条业务链按各自安全契约通过；草案必须通过 Verifier，缺信息或证据不足时
   必须安全追问/停止，不能编造结果。
3. 恢复诊断完整：32 条均成功注入故障并产生可审计记录；分别报告直接恢复率和恢复后的
   完整链成功率，不预先把未冻结阈值冒充 promotion 门。
4. 安全红线：出现越权动作、未经校验的行程、不可解释的重复工具调用或评测代码变化，均阻止晋升。

## 精确运行命令

修正后只运行一个入口，内部严格按“快照 → 测试 → 云端模型校验 → core → expanded →
recovery → 结束快照 → postflight”执行：

```powershell
& .\scripts\run_h006_full_agent_loop_eval.ps1 `
  -RunId h006-full-agent-loop-v1-20260904-r2 `
  -ExpectedManifestSha256 a374f4029cbac16c1bad63c99935385294e7bf485110bee797d5e7748ef955db `
  -RemoteHost connect.nmb1.seetacloud.com `
  -RemotePort 13793 `
  -RemoteUser root `
  -StageTimeoutSeconds 1800 `
  -MonitorIntervalSeconds 30
```

包装器显式创建唯一目录：
`artifacts/native-react-posttraining/h006-full-agent-loop-v1-20260904-r2/`。目录如果已存在则
立即拒绝运行；三个 evaluator 均不带 `--resume`，不会混入旧记录。

云端服务仍固定 checkpoint-32、LoRA rank 16、同一个 chat template 和 Hermes tool parser；
最大上下文从未经证明的 4096 提高到与训练合同一致的 6144。启动前先逐文件校验 base model、
checkpoint-32（包括 adapter config、tokenizer、权重和训练状态）、template 与四个关键包版本，
并检查 GPU 身份和空闲显存。启动后必须依次通过 `/health`、
`/v1/models` 中 LoRA 别名和一次 `enable_thinking=false + tool_choice=required` 的真实工具调用
smoke，才会进入正式评测。

意图解析的本地模型 fallback 在本轮进程中关闭。每条记录新增实际 `parse_source`、模型名和
backend；只要出现确定性 fallback、模型切换、HTTP 异常或被 evaluator 捕获的 exception，
该阶段立即判为基础设施无效，不再继续后续阶段。

本轮不向 LangSmith 上传轨迹。包装器会保存父进程中的新旧 `LANGSMITH_*` / `LANGCHAIN_*`
追踪与 API key 环境，给所有预检和评测子进程强制设置 tracing=false、key 为空，结束后再恢复。
manifest 已确认 traceable decorator 与 LangSmith SDK tracing 均为 false，两个 key 均不可见；
这只是隐私门禁，指标丢失的真正修复仍是让意图调用在异步边界内显式携带 metrics 返回。

## 监控与停止规则

- 顺序：完整运行快照 → 相关测试 → 模型/隧道/工具调用健康检查 → core → expanded → recovery →
  完整结束快照 → 机器 postflight。
- 每段使用独立隐藏进程并记录 PID、起止时间、初始/峰值 RSS、stdout、stderr 和真实退出码；
  每 30 秒把进程存活、日志字节数和内存写入 `monitor.jsonl`。GPU 只在启动前做有界门禁，
  不在硬超时主循环内发 SSH 请求。
- 单段评测硬超时 1800 秒；只有硬超时终止整个评测进程树并记录 exit 124。
- 连续 90 秒无输出只写 `OUTPUT_STALL_ADVISORY`，不自动重试、不自动改代码。
- 非零退出、报告条数不完整、模型身份不符、意图 fallback 或基础设施异常立即停止后续阶段。
- 模型业务失败但基础设施正常时继续收集剩余失败证据，不能用退出码 0 冒充模型通过。
- vLLM 在本轮独享的 session/process group 中启动；三段结束后先复核远端 PID、PGID、启动时钟
  和完整命令行，再整组关闭 API、engine 与 GPU worker。进程组为空、8000 端口释放、显存回落
  三项缺一不可；随后封口转录日志，最后由 postflight 对报告、三套完整 episode、日志及
  manifest 计算 SHA256。

## 审核问题修正映射

- 原来的 `Tee-Object` 父目录/退出码问题：改为包装器先建唯一目录，独立重定向 stdout/stderr，
  直接读取 native process exit code。
- 原来的超时和监控只有文字：现已实现为可执行逻辑。
- 原来只冻结 5 个文件：改为全 runtime surface 的逐文件双快照。
- 原来无法区分异常与业务失败：新增阶段级检查和最终 postflight。
- 原来 intent 模型可能静默切换：本地 fallback 禁用并逐 case 记录实际调用身份。
- 原来 evaluator 跨 LangSmith 异步边界读取可变 ContextVar，导致 r1 的 10 条 intent metrics
  全部丢失：现改为边界内读取并由 `IntentResult` 显式携带，普通、追问、修改、恢复及并发路径
  均有回归测试。
- 原来只关闭当前 LangSmith 变量：现同时强制关闭旧版 `LANGCHAIN_*` tracing 和 key，防止父环境
  的兼容变量绕过隐私门禁。
- 原来 recovery 用户身份可能与旧运行重合：新增强制唯一 `--rollout-id`。
- 原来的固定输出路径：改为唯一 Run ID，存在即失败，明确禁止 resume。
- 本次 wrapper 与 runtime manifest 双重硬锁 Run ID 为 r2；传 r1 或任意其他名字会在联网、建目录
  之前失败。
- 原来的 4096 上下文未证明：提高到训练预检使用的 6144。
- 原来 core case ID 校验写成统一 `fal-v2-*`：改为从冻结题库读取并逐条、按顺序精确比对。
- 原来普通、追问、revision 的 runtime error 字段不统一：现统一为 `runtime_errors`，同时兼容
  检查旧的单数字段，任何非空运行错误均使基础设施无效。
- 原来 GPU SSH 查询可能卡住硬超时：从超时主循环移除，循环只做本机有界检查；taskkill 后
  最多等待 10 秒，不再无限等。
- 所有独立 SSH 调用增加本地 wall-clock deadline：连通/身份检查 15 秒、清理 30 秒、完整远端
  checksum 预检 300 秒；云端连不上时在创建 Run ID 目录前失败，不消耗正式 Run ID。
- 原来 recovery 只有压缩 trace：现新增 32 条完整 EpisodeCandidate sidecar，保留每步完整
  observation，并校验每条恰有一次故障注入。
- 原来 revision 只收集修改后 episode：现将 initial 收集移到 revision 分支之前，并用回归测试
  强制每个 revision 同时保存 initial 与 revised 两条轨迹。
- 原来审核与执行之间没有机器锁：现用 789 文件 manifest 和命令行 SHA 双重锁定。
- manifest 中所有字符串行为配置只保存长度和 SHA256，凭据只保存“是否配置”；已清理 `.env`
  中误拼到普通字段的密钥片段并轮换本地 JWT secret，当前 manifest 不含敏感赋值明文。
- 原来只确认 API PID 和端口：现通过 `setsid` 建立本轮独享 PGID，清理后机器校验 PGID 无残留、
  端口释放、GPU 显存恢复，三项均纳入 postflight 硬门禁。
- 外部搜索、地图、数据库和缓存仍会随时间变化，所以本轮明确只算内部在线诊断；轨迹和
  工具 observation 全量落盘供复核，绝不据此宣称 promotion 通过。

## 下一状态

独立审核 Agent 在指标修复、隐私门禁、测试隔离与 RunId 双锁完成后给出本地 GO，无剩余 P0/P1。
正式九文件集合 98/98 passed，Ruff、Python compile 与两份 PowerShell AST 均通过。新 r2 manifest
已在隔离环境中生成并独立 verify 成功；其中 SDK tracing、decorator 和两个 key presence 均为 false，
敏感明文扫描通过。封口后已使用新端口 13793 再次完成最新只读远端预检：
SSH 可达，远端 8000 端口空闲；adapter/base combined hash、template hash 和四个包版本均与
冻结合同完全一致；GPU 为 RTX 4080 SUPER，总显存 32760 MiB，检查时空闲 32230 MiB。

用户确认精确命令后，r1 于 18:05:12 启动。789 文件 manifest、开始快照、87 项测试、远端模型
完整哈希、服务健康、LoRA 别名和真实工具调用 smoke 全部通过。core 10 于 18:18:34 完成，
进程退出码为 0、无超时、无 stall；原始诊断为 5/10，通过率 50%，但全部 10 条记录的
`intent_inference` 都是空值，因此无法从每条记录证明实际模型名和 backend。包装器按合同立即
阻止 expanded/recovery，并将本轮判为基础设施无效；该 50% 只能用于排查，不能作为正式成绩。

根因已用最小复现实验确认：启用 LangSmith `@traceable` 后，被装饰的异步调用运行在独立
ContextVar 上下文；意图模型在子上下文写入 `llm.last_request_metrics`，外层 evaluator 在
`process_user_turn` 返回后读取时得到 `None`。修复方向是让意图调用在装饰器边界内显式携带
metrics 返回，而不是跨边界读取可变 ContextVar，并增加 LangSmith 启用状态的回归测试。

失败后的清理门禁已通过：远端 PID/PGID 身份一致，进程组为空，8000 端口释放，GPU 显存从
基线 1 MiB 回落到 1 MiB。r1 目录永久保留，不覆盖、不续跑。修复、测试、重新审核并生成新
runtime manifest 后，下一次正式运行必须使用新的 Run ID `h006-full-agent-loop-v1-20260904-r2`。

正式入口自身也会先验证 789 文件 manifest，再做 15 秒 SSH 连通门禁；两项都发生在正式目录
创建之前。正式启动时把 Verification Status 改为 RUNNING；三段收集并审计后再改为
COMPLETED、FAILED、TIMED_OUT 或 INVALID_INFRASTRUCTURE。无论内部诊断结果如何，本轮
`production_promotion_eligible` 固定为 false。

### r2 实际结果

用户确认精确命令后，r2 完整执行了 preflight、core、expanded 和 recovery，三个评测进程均
exit 0、无超时、无 stall。98 项预检测试通过，模型健康、LoRA 别名和真实工具调用 smoke 通过。

- core：6/10，60%；基础设施有效，10/10 都是 `llm` 解析并记录正确 intent 模型/backend。
- expanded：13/20，65%；基础设施有效，20/20 intent 身份正确。
- recovery 原始诊断：15/32 严格通过，pass rate 46.88%；first-try recovery rate 50%，
  full-chain pass rate 50%。这三个恢复数字不能当正式有效成绩，因为 recovery 基础设施合同未满足。
- recovery 中 16/32 出现策略运行错误：9 条一次返回两个工具调用，7 条在失败后无进展地精确
  重复同一动作；8/32 因 Agent 在目标工具前失败而未成功注入故障。
- recovery 分层诊断：change/diagnostic 2/8，change/explicit 3/8，retry/diagnostic 4/8，
  retry/explicit 6/8。显式 retry 最稳，诊断后换参数最弱。
- 全部 62 条记录的 intent 均为 `parse_source=llm`、模型 `deepseek-v4-flash`、backend
  `cloud-openai-compatible`，证明 r1 的指标丢失问题已修复。

recovery 阶段完成后，包装器在阶段硬门禁发现非空 `runtime_errors`，立即停止 postflight 主路径并
按合同不重试。`finally` 清理验证通过：远端 PID/PGID 进程组为空、8000 端口释放、GPU 显存从
1 MiB 回落到 1 MiB。随后只补做结束快照和机器 postflight 以封存证据，没有重跑任何样本：
runtime combined SHA 前后均为
`50ec0ab50b13b83fb4348adc6505b153f6a8188b15aefbf7ec1edf418dfc66a3`；run contract、三段进程、
监控、artifact 完整性、远端身份和清理均有效；core/expanded episode sidecar 有效；recovery
sidecar 因部分 episode 无步骤或没有恰好一次注入而无效。最终机器状态为
`INVALID_INFRASTRUCTURE`，禁止 promotion，也禁止不分析原因就原样重跑。

### 独立专家复核与下一步门禁

独立审核 Agent 复核 postflight、三份报告和必要 episode 后确认，r2 必须保持
`INVALID_INFRASTRUCTURE`，不能事后修改冻结口径。core 6/10、expanded 13/20 和 62/62 intent
身份正确可作为“r2 内部诊断中的有效阶段结果”引用，但不能宣称 H006 通过或用于 promotion；
也不能另行拼成未预注册的 19/30 综合分。recovery 15/32、first-try 16/32、full-chain 16/32
只能用于诊断，因为 8/32 未进入故障条件且 episode sidecar 门禁失败。

复核认为观察到的失败首先是模型策略问题：一次生成两个 tool call，或在失败后无进展地原样
重复动作；不是 SSH、vLLM、intent 模型或外部工具宕机。与此同时，r2 的 evaluator 将模型协议
违例也写进 `runtime_errors`，再由总门禁统一当基础设施故障，分类合同不够精细。r3 应先完成：

1. 将 `PolicyOutputError` 单独记为 `model_policy_failure`，provider/transport、硬超时、注入器异常、
   数据损坏才进入 `runtime_error`。
2. 将注入状态拆成 `injected_once`、`not_reached_due_to_policy_failure`、`injector_error`；只有第三类
   使基础设施无效。
3. 零 step 的策略失败也保存明确终止事件、实际 tool-call 数和脱敏动作摘要，不伪造执行 step。
4. 预先冻结注入到达率、端到端 strict pass、已注入样本条件 recovery 三项指标，并增加首次双
   tool call、故障后重复、注入器异常和 provider/transport 异常四类测试。
5. 使用新 Run ID、新 manifest 先跑少量定向 smoke；通过后才决定全量 r3。模型侧后续纠偏样本
   聚焦“一次只发一个 tool call”和“按错误码决定换参数或原参重试”。

当前门禁：Immediate rerun = NO-GO；Immediate training = NO-GO。先修 r3 失败分类和可观测性，
完成定向 smoke 后再训练，避免把评测混杂问题当成学习目标。
