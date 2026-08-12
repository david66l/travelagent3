# Phase 0 工程基线报告（2026-08-12）

## 结论

TravelAgent2 已建立可回退、可重复执行的 Agentic RL 改造前基线。当前核心
Agent、LangGraph 编排、规划器和不依赖 PostgreSQL 的单元测试全部通过，可以
进入 Phase 1 的统一工具契约与 Validator 开发。

## 版本基线

- 开发分支：`feat/agentic-rl-long-horizon`
- 改造前快照：`1f12668 chore: freeze pre-agentic-rl development baseline`
- 本地配置、临时渲染文件和论文 PDF 均不进入 Git。
- 研究结论、论文引用索引和融合 SPEC 进入 Git。

## 已验证结果

| 门禁 | 结果 | 说明 |
|---|---:|---|
| Backend Ruff lint | 通过 | `src` 与 `tests` 无 lint 错误 |
| Backend Ruff format | 通过 | 已统一格式基线 |
| Backend MyPy | 通过 | 使用项目既有严格度配置 |
| Backend compileall | 通过 | 源码可编译 |
| 离线单元测试 | 606 passed | 不需要 PostgreSQL/Redis/真实 LLM |
| 数据库单元测试 | 123 deselected | 已标记 `requires_db`，服务启动后执行 |
| 核心 Graph/Planner 测试 | 84 passed | 独立定向验证 |
| Frontend TypeScript | 通过 | `tsc --noEmit` |
| Frontend production build | 环境阻断 | `next/font` 拉取 Google Fonts 超时 |
| Gateway | 未验证 | 当前 Windows 环境没有 Go 工具链 |
| Docker 集成链 | 未验证 | Docker Desktop 当前未运行 |

## 本轮修复的真实缺陷

1. 修复旧 SSE 端点引用不存在的 `agent_runner`，统一使用 `graph_runner`。
2. 修复 Celery 最终失败路径漏导入 `push_dead_letter`。
3. 修复用户画像中 `has_children=None` 导致 Pydantic 校验失败。
4. 幻觉检测在没有验证证据时改为“通过但得分 0”，避免把未验证结果记成满分，
   污染后续 SFT/GRPO 轨迹与 Reward。
5. 价格查询增加确定性降级；外部检索最多等待 1 秒，失败后不会返回空价格。
6. 恢复 WebSocket `send_json` 契约，同时保留不可序列化对象的安全归一化。
7. 天气建议函数补回默认降水概率，兼容已有调用契约。

## 重复执行

在仓库根目录使用后端虚拟环境运行：

```powershell
backend\.venv\Scripts\python.exe scripts\check_phase0.py
```

PostgreSQL 可用时执行完整单元测试：

```powershell
backend\.venv\Scripts\python.exe scripts\check_phase0.py --with-db
```

CI 或工具链完整的开发机可追加 `--strict-toolchains`，要求前端和 Gateway 也不得
跳过。

## 尚未解除的外部阻塞

- 启动 Docker Desktop 后，运行 123 个数据库测试以及 Docker 集成测试。
- 安装 Go 1.22+ 后，验证 Gateway 的 lint、test 和 build。
- 将 `next/font` 改为本地字体或稳定构建资源，解除生产构建对 Google Fonts 的依赖。
- 正式 SFT/GRPO 需要真实数据集、模型权重许可和适合的 GPU 资源；本机 8GB GPU
  主要承担 QLoRA、小规模训练验证与推理。

## 下一阶段入口

Phase 1 先实现共享的 `Observation Envelope`、工具错误语义和程序化 Validator，
再接入 Goal Ledger 与 Task DAG。Reward 只消费经过版本化验证器计算出的事实指标，
避免在线逻辑、训练环境和评测口径彼此割裂。
