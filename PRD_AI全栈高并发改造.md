# TravelAgent2 高并发 AI 全栈改造 PRD

> 版本：v1.0  
> 日期：2026-06-15  
> 状态：待评审 / 待开发  
> 目标：将当前「单进程异步演示版」旅行 Agent 改造为可水平扩展、具备自研模型服务、完整安全与成本治理的 AI 全栈高并发系统。

---

## 1. 项目背景与目标

### 1.1 当前基线

TravelAgent2 当前已实现：
- 中文多轮对话收集旅行需求
- 意图识别 + 实体抽取
- POI / 天气 / 价格查询（含内置 fallback）
- 确定性日程编排 + 规则校验 + 修复引擎
- LLM 文案润色
- 基于 DB lease 的异步 Worker
- WebSocket 实时阶段推送
- Redis 会话热缓存 + PostgreSQL 持久化

### 1.2 核心短板

| 维度 | 当前状态 | 高并发下的风险 |
|---|---|---|
| 接入层 | 无认证、CORS 全开、无限流 | 易被刷接口、无法区分用户权限 |
| 实时通道 | WebSocket 有状态、单进程连接池 | 水平扩展困难、连接无法漂移 |
| 任务调度 | 单 Worker DB 轮询 | 吞吐量受限、无法多实例消费 |
| 模型层 | 同步调用第三方 API、无流式 | 延迟高、成本高、无法自主扩缩 |
| 工具层 | 天气/价格为模拟、无熔断降级 | 外部抖动直接影响可用性 |
| 缓存 | 仅基础 Redis 缓存、无预热/防雪崩 | 缓存命中率低、容易穿透 |
| 可观测性 | 本地日志、无 metrics | 无法定位瓶颈、无法容量规划 |

### 1.3 改造目标

**一句话目标**：构建「Go 网关 + 无状态 FastAPI + Celery 异步队列 + vLLM 自研推理 + Redis 共享状态 + K8s 弹性伸缩」的高并发 AI 全栈旅行规划系统。

**量化目标**：
- 单实例支持 ≥ 1000 并发 WebSocket/SSE 连接
- 行程生成任务支持水平扩展，单集群 ≥ 10 worker
- 首次行程草稿返回 < 500ms（缓存命中）/ < 3s（完整生成）
- 外部服务故障时 100% 可降级到内置数据
- 简单问答 Token 成本降低 60%（大小模型路由）
- 系统可用性 ≥ 99.5%（含外部依赖降级）

---

## 2. 术语表

| 术语 | 说明 |
|---|---|
| SSE | Server-Sent Events，服务器单向推送流，基于 HTTP |
| vLLM | 开源 LLM 推理引擎，支持 continuous batching 和 LoRA adapter |
| LoRA | Low-Rank Adaptation，轻量级模型微调方法 |
| Celery | Python 分布式任务队列 |
| Worker | 消费异步任务的后台进程 |
| Gateway | Go 编写的 API 网关，负责鉴权、限流、路由 |
| POI | Point of Interest，兴趣点（景点、餐厅、酒店等） |
| TTFT | Time To First Token，首 Token 返回时间 |
| HPA | Kubernetes Horizontal Pod Autoscaler |

---

## 3. 总体架构

### 3.1 部署架构

```
                              ┌─────────────────┐
                              │   CDN / WAF     │
                              └────────┬────────┘
                                       │
                              ┌────────▼────────┐
                              │   Go Gateway    │  JWT / 限流 / 熔断 / WAF
                              │   (Replicas=3)  │
                              └────────┬────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
┌───────▼────────┐           ┌─────────▼──────────┐        ┌──────────▼─────────┐
│  Next.js SSR   │           │  FastAPI Services  │        │  Celery Workers    │
│  (Replicas=2)  │           │  (Replicas=3+)     │        │  (Replicas=5+)     │
└────────────────┘           └─────────┬──────────┘        └──────────┬─────────┘
                                       │                              │
                              ┌────────▼────────┐           ┌─────────▼────────┐
                              │   Redis Cluster │           │   vLLM Cluster   │
                              │   (3主3从)      │           │   · 基础模型     │
                              │   · 缓存/状态   │           │   · LoRA adapter │
                              │   · 限流/队列   │           │   · 流式推理     │
                              │   · Pub/Sub     │           │   · 推理路由     │
                              └────────┬────────┘           └──────────────────┘
                                       │
                              ┌────────▼────────┐
                              │   PostgreSQL    │
                              │   · 用户/行程   │
                              │   · 任务事件    │
                              │   · 冷记忆归档  │
                              └─────────────────┘
```

### 3.2 数据流

1. 用户通过 Next.js 前端建立 SSE 连接（`/api/v1/chat/stream`）
2. Go Gateway 校验 JWT，解析用户身份，进行 QPS 限流
3. 请求转发到 FastAPI，FastAPI 将状态读/写 Redis
4. 若需行程生成，提交 Celery 任务，立即返回 `job_id`
5. Celery Worker 执行规划 pipeline，阶段结果写入 Redis Pub/Sub
6. FastAPI 通过 SSE 将阶段/Token 流推送给前端
7. 完整行程落库 PostgreSQL，会话状态按策略归档

### 3.3 技术选型

| 层级 | 选型 | 理由 |
|---|---|---|
| 网关 | Go + Echo/Gin | 高并发、低延迟、编译型语言适合流量入口 |
| Web 框架 | FastAPI（保留） | 已有基础，异步原生，生态成熟 |
| 任务队列 | Celery + Redis | 成熟稳定，支持优先级、重试、死信 |
| 模型服务 | vLLM | 高吞吐、支持 LoRA、OpenAI-compatible API |
| 缓存/状态 | Redis Cluster 6.x | 低延迟、支持 Pub/Sub、数据结构丰富、原生集群分片 |
| 数据库 | PostgreSQL（保留） | 已有基础，JSONB 适合存储对话/行程 |
| 前端 | Next.js 15（保留） | 已有基础，支持 SSR 和流式消费 |
| 部署 | Kubernetes | 弹性伸缩、服务发现、配置管理 |
| 可观测 | Prometheus + Grafana + Loki | Metrics + Logs 一体化 |
| 模型实验 | MLflow | LoRA 版本管理、A/B 灰度 |

---

## 4. 功能需求

### 4.1 接入层：Go 网关

#### 4.1.1 JWT 鉴权

- 支持游客模式（guest token，有效期 24h，仅基础聊天）
- 支持登录用户（access token + refresh token，保存历史行程、偏好）
- Token 从 `Authorization: Bearer <token>` 读取
- WebSocket/SSE 握手时同样校验 token

##### Token 吊销机制

- **用户登出**：
  - 登录用户的 access token 和 refresh token 加入 Redis 黑名单
  - Key：`jwt_blacklist:{token_hash}`，TTL = token 剩余有效期
  - 后续请求解析 token 时，先查黑名单，命中则返回 401
- **账号异常**：
  - 管理员封禁用户时，将该用户所有已签发 token 加入黑名单
  - 通过 `jwt_blacklist:user:{user_id}` Set 记录该用户所有生效 token 的 hash，封禁时全部标记
- **Token 刷新**：
  - 使用 refresh token 换取新 access token 后，旧 access token 可选加入黑名单（默认加入，TTL 为原 token剩余时间）

##### 游客 Token 唯一性校验

- 游客 token 与 **设备指纹** 绑定
- 设备指纹生成：`sha256(user_agent + screen_resolution + timezone + canvas_fingerprint)`，前端通过 header `X-Device-Fingerprint` 传递
- 网关校验：
  - 解析 guest token 中的 `device_fingerprint` 声明
  - 与请求头中的设备指纹比对，不一致则返回 403
  - 同一 token 在多个设备使用时，仅允许绑定设备访问
- 游客 token 禁止刷新，过期后需重新获取

##### Token 异常响应与前端降级

| 场景 | HTTP 状态码 | 响应体 | 前端处理 |
|---|---|---|---|
| Authorization 缺失 | 401 | `{"code": "AUTH_MISSING", "message": "请先登录"}` | 跳转登录页 / 显示游客模式入口 |
| Token 过期 | 401 | `{"code": "TOKEN_EXPIRED", "message": "登录已过期，请重新登录"}` | 尝试 refresh token，失败则跳转登录 |
| Token 无效/解析失败 | 401 | `{"code": "TOKEN_INVALID", "message": "登录状态异常"}` | 清除本地 token，跳转登录 |
| Token 被吊销 | 401 | `{"code": "TOKEN_REVOKED", "message": "登录状态已失效"}` | 清除本地 token，跳转登录 |
| 游客 token 设备不一致 | 403 | `{"code": "DEVICE_MISMATCH", "message": "请在原设备继续使用或重新获取游客身份"}` | 重新获取 guest token |
| 权限不足 | 403 | `{"code": "FORBIDDEN", "message": "当前身份无法使用该功能"}` | 提示登录升级 |

- 前端统一拦截 401/403，自动清理失效 token 并跳转登录/游客入口
- SSE/WebSocket 连接因 token 异常断开时，前端不自动无限重连，而是弹出登录提示

#### 4.1.2 权限模型

| 角色 | 权限 |
|---|---|
| 游客 | 多轮对话、生成 1 次行程、不能保存历史 |
| 登录用户 | 无限对话、保存行程、查看历史、设置偏好 |
| 管理员 | 查看系统指标、手动切换模型版本 |

#### 4.1.3 限流策略

| 维度 | 策略 | 阈值（可配置） |
|---|---|---|
| IP | 滑动窗口限流 | 60/min |
| 用户 | 滑动窗口限流 | 30/min |
| 游客 | 更严格限流 | 10/min |
| SSE 连接 | 单用户最大并发连接数 | 3 |
| LLM Token | 单用户日配额 | 游客 10k / 用户 100k |

#### 4.1.4 熔断与降级

- 对 FastAPI 后端配置熔断器（失败率 > 50% 且 10s 内错误 > 20 次触发）
- 熔断后返回 503 + 友好提示，并触发告警
- 支持按路径细粒度熔断（如 `/api/v1/itinerary` 独立熔断）

#### 4.1.5 路由转发

| 路径 | 目标 | 说明 |
|---|---|---|
| `/api/v1/*` | FastAPI | 业务 API |
| `/api/v1/chat/stream` | FastAPI SSE | 流式对话 |
| `/ws/v1/chat` | FastAPI WebSocket（可选保留） | 双向实时通道 |
| `/health` | Gateway 自身健康 | K8s 探针 |
| `/metrics` | Prometheus metrics | 网关指标 |

### 4.2 实时通道：SSE 为主，WebSocket 为辅

#### 4.2.1 SSE 流式对话

- 端点：`GET /api/v1/chat/stream?session_id=xxx`
- 事件类型：
  - `message`：用户/助手消息
  - `stage`：任务阶段变更（intent / data / draft / validate / writing / completed）
  - `token`：LLM 生成 Token 流
  - `job`：任务创建/更新
  - `error`：错误信息
  - `done`：流结束

#### 4.2.2 WebSocket 保留场景

- 仅用于需要双向低延迟交互的场景（如语音输入、实时协同编辑）
- 若保留，需通过 Redis Pub/Sub 实现多节点消息广播
- 默认推荐：**SSE 满足所有文本对话场景，WebSocket 暂不启用**

### 4.3 会话与记忆体系

#### 4.3.1 双层记忆架构

| 层级 | 存储 | TTL / 策略 | 用途 |
|---|---|---|---|
| 热记忆 | Redis | 30min，每次访问续期 | 当前会话上下文、最近消息 |
| 温记忆 | Redis | 24h | 当日活跃用户 profile 快照 |
| 冷记忆 | PostgreSQL | 永久 | 历史会话、用户偏好、归档消息 |

#### 4.3.2 记忆持久化与归档

- 活跃会话每 5 分钟异步归档到 PostgreSQL
- 会话结束（断开 30min 后）触发最终归档
- 归档后 Redis 保留 24h 用于快速恢复
- 超过 90 天的冷记忆做压缩存储（仅保留摘要）

#### 4.3.3 过期清理

- Redis Key 统一设置 TTL
- 每日凌晨运行清理任务，删除过期归档
- PostgreSQL 按时间分区（`conversation_archives_YYYYMM`）

#### 4.3.4 高并发一致性

- 会话状态写入使用 Redis 乐观锁（`WATCH` + `MULTI/EXEC`）
- 或使用 Redis Hash + 版本号字段，写时校验
- **核心会话写操作使用 Redlock 分布式锁**（多主 Redis Cluster 场景）
  - 锁 key：`lock:session:{session_id}:write`
  - 锁 TTL：5s，业务完成后立即释放
  - 获取锁超时：100ms，避免长时间阻塞
- 多设备同时访问同一会话时，获取锁成功后写入，失败端返回冲突提示

### 4.4 Agent 工作流

#### 4.4.1 流程阶段（保留并增强）

```
1. intent_ready      → 意图识别 + 实体抽取
2. data_collection   → POI / 天气 / 价格并行查询（异步 + 缓存）
3. draft_ready       → 启发式策略 + 确定性日程编排
4. itinerary_final   → 规则校验 + Repair Loop
5. writing           → LLM 润色文案（流式输出）
6. completed         → 返回最终结果
```

#### 4.4.2 确定性修复引擎增强

新增异常场景识别与自愈策略：

| 异常场景 | 检测规则 | 修复动作 |
|---|---|---|
| 节假日满房 | 酒店 POI 返回空/高价告警 | 扩大搜索半径、推荐替代住宿区域 |
| 路线冲突 | 两地距离 > 30km 且同半天安排 | 拆分至不同天、或插入交通枢纽 |
| 跨城市规划 | 用户要求多日跨城 | 自动加入城际交通节点、校验可行性 |
| 预算超限 | 总费用 > 预算 120% | 按优先级删减非必去项目 |
| 时间重叠 | 两个活动时段冲突 | 后移、缩短、删除 |
| 营业时间不匹配 | 活动安排在闭馆时段 | 调整顺序或替换 |

修复循环：最多 10 轮，超过则标记 `needs_clarification` 转人工/用户确认。

### 4.5 工具层

#### 4.5.1 工具统一框架

- 所有外部工具实现统一接口：`Tool.execute(params) -> ToolResult`
- 支持：同步/异步执行、超时、重试、降级、缓存、熔断

#### 4.5.2 POI 查询

##### 数据源优先级

| 数据源 | 优先级 | 降级策略 |
|---|---|---|
| 内置城市数据 | 1 | 零延迟、确定性 |
| 高德/百度 Places API | 2 | 失败/超时回退到内置数据 |
| Tavily + LLM 提取 | 3 | 网络搜索兜底 |

##### 部分失败降级

- POI 查询按景点类型拆分为多个子查询（如「景点」「餐厅」「酒店」「购物」）
- 每个子查询独立执行、独立降级：
  - 某类 POI 查询失败时，仅该类降级为内置数据，不影响其他类
  - 核心 POI（如城市地标、5A 景区）允许 3 次重试 + 2 次数据源降级
  - 非核心 POI（如小众咖啡馆）允许 1 次重试，失败后直接降级
- 返回结果包含 `data_source` 字段：
  - `api`：真实接口返回
  - `built_in`：内置兜底数据
  - `fallback`：简化估算数据
  - `unavailable`：无法获取

##### POI 优先级分类

| 优先级 | 类型 | 示例 | 重试策略 | 降级策略 |
|---|---|---|---|---|
| P0 核心 | 地标/5A 景区/必去 | 西湖、故宫 | 3 次重试 | API → 内置 → 网络搜索 |
| P1 重要 | 热门餐厅/特色体验 | 米其林餐厅 | 2 次重试 | API → 内置 |
| P2 普通 | 一般景点/购物 | 商场、公园 | 1 次重试 | API → 内置 |
| P3 可选 | 小众/备选 | 咖啡馆、书店 | 0 次重试 | 直接内置 |

##### 数据标注与前端展示

- 每个 POI 返回结构：
  ```json
  {
    "id": "poi_xxx",
    "name": "西湖",
    "data_source": "api",
    "confidence": 0.95,
    "is_fallback": false,
    "fallback_reason": null
  }
  ```
- 前端根据 `data_source` 展示不同标识：
  - `api`：正常展示
  - `built_in`：标注「基于本地推荐」
  - `fallback`：标注「数据可能不准确，建议核实」
  - `unavailable`：标注「暂无数据」

- 批量查询拆分为并发子任务，单任务 3s 超时，指数退避重试
- 结果缓存 6h，按 `poi:{city}:{category}:{query_hash}` 命名

#### 4.5.3 天气查询

- 接入真实天气 API（如和风天气/OpenWeather）
- 缓存 1h
- 失败时返回「暂无实时天气，建议出行前查看」占位文案

#### 4.5.4 价格查询

- 接入真实价格数据源（酒店 OTA、门票平台）
- 缓存 30min
- 失败时返回价格区间估算

#### 4.5.5 路线/距离计算

- 接入高德/百度 Distance Matrix API
- 批量异步计算，缓存 24h
- 失败时回退到简化平面距离公式

### 4.6 模型层

#### 4.6.1 模型服务架构

```
vLLM Cluster
├── Base Model（如 Qwen2.5-14B-Instruct / DeepSeek-V2.5）
├── LoRA Adapter: travel-chat-v1      → 简单问答、意图识别
├── LoRA Adapter: travel-plan-v1      → 行程生成、润色
└── LoRA Adapter: travel-repair-v1    → 异常解释、用户沟通
```

#### 4.6.2 LoRA 微调

- 数据来源：
  - 自有历史对话数据（清洗后）
  - 合成数据：基于模板生成多轮对话 + 行程对
  - 用户反馈：点赞/修改记录回流
- 数据格式：ShareGPT / Alpaca 格式
- 训练任务：
  - `ml/training/prepare_data.py`
  - `ml/training/train_lora.py`
  - `ml/training/evaluate.py`
  - `ml/training/merge_and_upload.py`

#### 4.6.3 大小模型混合调度

| 任务类型 | 模型 | 说明 |
|---|---|---|
| 意图识别 | 小模型（7B） | 低延迟、低成本 |
| FAQ / 简单问答 | 小模型（7B） | 通用知识 |
| 行程生成 | 大模型（14B+）+ LoRA | 复杂推理 |
| 文案润色 | 大模型（14B+）+ LoRA | 生成质量 |
| 异常解释 | 中模型（7B-14B） | 平衡质量与成本 |

路由决策由 `core/model_router.py` 根据：
- 任务类型
- 上下文长度
- 用户等级
- 当前队列负载

### 4.7 异步任务系统

#### 4.7.1 Celery 任务定义

- `tasks.itinerary.generate_itinerary_task(job_id)`
- `tasks.poi.batch_poi_query_task(city, keywords, job_id)`
- `tasks.memory.archive_session_task(session_id)`
- `tasks.memory.cleanup_expired_memories_task()`

#### 4.7.2 任务状态机

```
PENDING → RUNNING → COMPLETED
   ↓         ↓
CANCELLED  FAILED → RETRYING → RUNNING
```

#### 4.7.3 重试策略

| 任务类型 | 初始间隔 | 最大退避 | 最大重试次数 | Jitter | 重试条件 |
|---|---|---|---|---|---|
| 行程生成 | 2s | 30s | 3 | True | 外部 API 超时、LLM 暂时不可用 |
| POI 批量查询 | 1s | 10s | 3 | True | 搜索 API 限流/超时 |
| 记忆归档 | 5s | 60s | 5 | True | 数据库连接异常 |
| 记忆清理 | 10s | 120s | 5 | True | 数据库连接异常 |

- 配置示例：`retry_backoff=2, retry_backoff_max=30, retry_jitter=True, max_retries=3`
- 非重试异常：参数错误、权限错误、用户主动取消，直接标记 FAILED
- 重试计数持久化到 Redis，避免 Worker 重启丢失

#### 4.7.4 死信队列（Dead Letter Queue）

- **队列名**：`planning_dead_letter`
- **入队条件**：任务重试耗尽后自动入队
- **死信消息格式**：
  ```json
  {
    "task_id": "uuid",
    "task_name": "generate_itinerary_task",
    "job_id": "uuid",
    "failed_at": "2026-06-15T00:00:00Z",
    "exception": "TimeoutError",
    "traceback": "...",
    "args": [],
    "kwargs": {}
  }
  ```
- **处理机制**：
  - 每日凌晨 Celery Beat 触发死信巡检任务
  - 生成《失败任务日报》并推送告警（企业微信/钉钉/邮件）
  - 提供管理后台接口供运维人员手动重试或标记忽略
  - 超过 7 天未处理的死信自动归档到 PostgreSQL `dead_letter_archive`

#### 4.7.5 任务取消机制

- **取消入口**：`POST /api/v1/jobs/{id}/cancel`
- **状态流转**：`RUNNING → CANCELLING → CANCELLED`
- **实现逻辑**：
  1. API 接收到取消请求后，将 `job:cancel:{job_id}` 发布到 Redis Pub/Sub
  2. 任务执行过程中，每完成一个子阶段前检查 Redis 是否存在取消信号
  3. 若检测到取消信号，保存已完成的中间结果，更新状态为 `CANCELLED`，清理资源
  4. 若任务已处于最后阶段（如 writing），允许完成当前阶段后再取消，避免数据不一致
- **强制终止**：
  - 发送取消信号 30s 后任务仍未停止，标记为 `FORCE_CANCELLED`
  - 由独立监控任务检查并清理残留 Worker 进程
- **优雅取消支持点**：
  - POI 查询批次之间
  - 规则校验/修复循环之间
  - LLM 调用前后（不中断已发送的 LLM 请求，但不再处理后续阶段）

#### 4.7.6 队列划分

| 队列 | 用途 | Worker 数量 |
|---|---|---|
| `default` | 普通任务 | 3 |
| `planning` | 行程生成 | 5+ |
| `poi` | POI 批量查询 | 3 |
| `memory` | 记忆归档/清理 | 1 |
| `planning_dead_letter` | 死信任务 | 0（仅消费巡检任务） |

### 4.7.4 Redis Cluster 部署规范

#### 节点与分片

- **拓扑**：3 主 3 从，共 6 个节点，每主节点 1 从
- **分片规则**：
  - 默认 16384 个 slot，每主节点负责约 5461 个 slot
  - 使用 Redis Cluster 原生哈希槽分片，key 的 slot = CRC16(key) % 16384
- **部署模式**：StatefulSet + Headless Service，每个 Pod 独立 DNS

#### 持久化策略

- **AOF**：开启 `appendonly yes`，策略 `appendfsync everysec`
  - 保证最多丢失 1 秒数据
  - AOF 重写策略：`auto-aof-rewrite-percentage 100`，`auto-aof-rewrite-min-size 64mb`
- **RDB**：保留 RDB 快照，`save 900 1 / save 300 10 / save 60 10000`
- **混合模式**：AOF + RDB 同时启用，RDB 用于快速全量恢复，AOF 用于增量恢复

#### 内存与淘汰策略

- **最大内存**：每节点 `maxmemory 8gb`（根据实际机器调整）
- **淘汰策略**：`maxmemory-policy volatile-lru`
  - 仅淘汰设置了 TTL 的 key
  - 优先淘汰最近最少使用的 key
- **大 Key 限制**：单个 key 大小不超过 1MB，list/hash/set/zset 元素数不超过 5000

#### 高可用与容灾

- **故障检测**：
  - 节点间互相 PING/PONG，心跳间隔 1s
  - 主观下线（PFAIL）：`node-timeout` 内未收到回复
  - 客观下线（FAIL）：多数主节点确认 PFAIL
- **故障切换**：
  - 主节点客观下线后，从节点自动发起选举
  - 集群可用条件：至少半数主节点存活，且每个不可用的主节点至少有一个从节点存活
- **数据备份**：
  - 每日凌晨将 RDB 备份到对象存储
  - 跨区域容灾：主集群 + 只读从集群（异步复制）

#### 热点 Key 规避

- **Key 设计**：
  - 避免大量请求集中在同一个 key
  - 对极热 key 加随机后缀分片：`poi:hot:shanghai:0` ~ `poi:hot:shanghai:9`
- **本地缓存**：在 FastAPI 实例内加 L1 缓存（如 `cachetools.TTLCache`），减少 Redis 热 key 压力
- **读写分离**：从节点处理读请求，主节点处理写请求

#### 连接池配置

- 每 FastAPI 实例 Redis 连接池：
  - 最大连接数：100
  - 最小空闲连接：10
  - 连接超时：2s
  - 读取超时：3s

---

### 4.8 缓存策略

#### 4.8.1 缓存分层

| 层级 | 内容 | TTL |
|---|---|---|
| L1 本地缓存 | 热门城市 POI、模型路由规则 | 5min |
| L2 Redis | POI、天气、价格、热门行程模板 | 30min–12h |
| L3 数据库 | 完整行程、用户历史 | 永久 |

#### 4.8.2 缓存 Key 规范

```
itinerary:{city}:{theme}:{days}:{budget_range}:{v}   → 热门行程模板
poi:{city}:{category}:{query_hash}:{v}               → POI 列表
weather:{city}:{date}:{v}                            → 天气
price:{poi_id}:{date}:{v}                            → 价格
rate_limit:{type}:{identifier}                       → 限流计数
session:{session_id}:state                           → 会话热状态
```

#### 4.8.3 缓存预热

- 启动时加载 Top-20 城市 POI 到 Redis
- 每日凌晨预热次日热门城市天气

#### 4.8.4 缓存保护

- 布隆过滤器防止缓存穿透
- 随机 TTL 防止缓存雪崩
- 缓存击穿：热点 Key 加互斥锁，单线程回源

### 4.9 AI 安全

#### 4.9.1 输入防护

##### 提示注入检测

- **规则层**：
  - 黑名单关键词（不区分大小写）：
    - `ignore previous instructions`
    - `ignore all prior instructions`
    - `DAN` / `Do Anything Now`
    - `jailbreak`
    - `you are now in developer mode`
    - `system prompt`
    - `role-play as`
    - 连续特殊符号或超长无意义输入
  - 命中规则直接拦截，返回 400

- **语义层**：
  - 使用轻量分类模型（如 fine-tuned BERT / small LLM）对输入进行注入风险评分
  - 阈值：置信度 ≥ 0.8 判定为提示注入，直接拦截
  - 0.5–0.8 之间标记为可疑，允许通过但记录日志并降低模型 temperature
  - < 0.5 视为正常

- **拦截响应**：
  ```json
  {
    "code": "PROMPT_INJECTION_DETECTED",
    "message": "检测到异常输入，请用自然的旅行需求描述重新提问。"
  }
  ```

- **敏感指令过滤**：
  - 禁止模型执行系统命令、泄露 prompt、修改系统角色设定
  - 对用户输入中的角色扮演请求进行拦截

#### 4.9.2 输出合规

##### 敏感内容过滤

- **敏感 POI / 地区过滤**：
  - 维护 `sensitive_locations` 字典，包含：
    - 政治敏感地点
    - 军事管制区域
    - 未开放/危险景区
    - 违法活动场所
  - LLM 输出中提到敏感地点时，自动替换为「该地点暂不适合推荐」或删除该条目

- **内容安全规则**：
  - 政治敏感、暴力、恐怖、歧视、色情内容 → 直接过滤并替换
  - 违法活动（如非法穿越、未开放区域探险）→ 删除并添加风险提示
  - 涉及危险户外活动（如悬崖、深海潜水、无人区）→ 保留但添加「请在专业指导下进行」提示

##### 事实校验

- Fact Guard 防止 LLM 篡改 POI 事实
- 输出中 POI 名称、票价、开放时间等字段必须来源于工具返回的数据，不得 LLM 编造

##### 风险提示规则

| 场景 | 处理方式 |
|---|---|
| 高原/登山活动 | 添加「注意高反/体力要求」 |
| 水上活动 | 添加「注意水上安全」 |
| 夜间出行 | 添加「注意夜间安全」 |
| 边境/偏远地区 | 添加「提前了解当地政策」 |
| 极端天气 | 添加「出行前关注天气」 |

#### 4.9.3 传输与存储加密

##### HTTPS 强制

- Go Gateway 拦截所有 HTTP 请求，301 重定向到 HTTPS
- 拒绝明文 HTTP 的业务请求（除健康检查外）
- HSTS 头配置：`Strict-Transport-Security: max-age=31536000; includeSubDomains`

##### Redis 加密

- **传输加密**：Redis Cluster 启用 TLS，所有客户端使用 `rediss://` 连接
- **静态加密**：
  - 敏感 key（如 session、token 黑名单）使用 AES-256 加密后存储
  - 密钥通过 K8s Secret 注入，定期轮换

##### PostgreSQL 加密

- **传输加密**：PostgreSQL 启用 SSL/TLS
- **字段级加密**：
  - 使用 `pgcrypto` 插件加密手机号、邮箱、身份证
  - 应用层加密 + 数据库层加密双重保护
- **备份加密**：所有备份文件加密后上传到对象存储

#### 4.9.4 安全扫描机制

| 扫描类型 | 频率 | 工具 | 处理流程 |
|---|---|---|---|
| 依赖漏洞扫描 | 每周 | Snyk / Dependabot | 高危漏洞 7 天内修复 |
| 代码安全扫描 | 每次 PR | Bandit / Semgrep | PR 阻塞修复 |
| 容器镜像扫描 | 每次构建 | Trivy / Clair | 高危漏洞禁止推送 |
| 密钥泄露扫描 | 每次 PR | GitLeaks / TruffleHog | 发现即回滚 |
| 渗透测试 | 每月 | 外部安全团队 / 自动化工具 | 生成报告并修复 |
| 基线配置扫描 | 每月 | CIS Benchmark | 不合规项限期整改 |

---

#### 4.9.5 隐私保护

##### 脱敏规则

- **身份证号**：保留前 6 位 + 后 4 位，中间用 `*` 替代，如 `110101********1234`
- **手机号**：保留前 3 位 + 后 4 位，中间用 `*` 替代，如 `138****1234`
- **邮箱**：保留首字母和域名，如 `u***@example.com`
- **银行卡号**：保留后 4 位，如 `**** **** **** 1234`
- **地址**：保留到省市区，详细门牌号用 `*` 替代
- **姓名**：保留姓氏，名用 `*` 替代，如 `张*`

##### 脱敏函数定义

```python
def mask_sensitive(text: str) -> str:
    """
    对文本中的敏感信息进行脱敏处理。
    支持：身份证、手机号、邮箱、银行卡、地址、姓名。
    """
    # 使用正则表达式识别并替换
    ...
```

- **处理位置**：
  - 用户输入进入系统前先做脱敏，原始内容不入日志
  - LLM 看到的上下文为脱敏后内容
  - 第三方 API 调用前对参数脱敏
  - 数据库中存储原始内容（加密），日志中仅存储脱敏后内容

##### 日志隐私

- 日志中不打印完整会话内容
- 用户 ID、会话 ID 可记录，但消息内容脱敏后记录
- 第三方 API 调用日志中移除 API Key 和敏感参数

### 4.10 成本优化

#### 4.10.1 Token 成本控制

- 提示词压缩：长对话自动摘要，保留关键约束
- 缓存优先：热门请求直接命中缓存
- 模型分层：小模型处理 70% 简单请求

#### 4.10.2 GPU 成本控制

- vLLM continuous batching 提高吞吐
- 夜间低谷期缩容 vLLM pod
- 冷热模型分离：不常用 LoRA 按需加载

#### 4.10.3 外部 API 成本控制

- POI 查询优先内置数据
- 天气/价格缓存，避免频繁调用
- 搜索 API 用量配额监控

#### 4.10.4 成本监控指标

| 指标名称 | 类型 | 说明 | 标签 |
|---|---|---|---|
| `llm_token_usage_total` | Counter | LLM Token 总消耗量 | `model`, `task_type`, `user_tier` |
| `llm_request_duration_seconds` | Histogram | LLM 请求耗时 | `model`, `task_type` |
| `vllm_gpu_utilization` | Gauge | vLLM GPU 利用率 | `pod`, `gpu` |
| `vllm_gpu_memory_used_bytes` | Gauge | vLLM GPU 显存使用 | `pod`, `gpu` |
| `external_api_calls_total` | Counter | 外部 API 调用次数 | `api_name`, `status`, `data_source` |
| `external_api_cost_total` | Counter | 外部 API 预估费用（元） | `api_name` |
| `celery_task_cost_total` | Counter | 任务执行成本（元） | `task_name` |

#### 4.10.5 成本阈值告警

| 告警名称 | 条件 | 级别 | 通知方式 |
|---|---|---|---|
| LLM Token 日消耗过高 | `sum(llm_token_usage_total{day="today"}) > 10,000,000` | Warning | 钉钉/企业微信 |
| LLM Token 日消耗超限 | `sum(llm_token_usage_total{day="today"}) > 30,000,000` | Critical | 电话 + 钉钉 |
| 外部 API 日调用过高 | `sum(external_api_calls_total{day="today"}) > 50,000` | Warning | 钉钉 |
| GPU 利用率持续高位 | `vllm_gpu_utilization > 90% 持续 10min` | Warning | 钉钉 |
| 单用户 Token 异常 | `sum by (user_id)(llm_token_usage_total) > 100,000 / 天` | Warning | 钉钉 |

#### 4.10.6 成本熔断开关

- **触发条件**（满足任一）：
  - 单日 LLM Token 消耗 > 成本熔断阈值（如 5000 万 Token）
  - 单日外部 API 调用费用 > 熔断阈值（如 1000 元）
  - 单小时 GPU 成本 > 熔断阈值（如 500 元）
- **熔断动作**：
  - 自动将复杂规划请求路由到轻量模型
  - 优先使用缓存/内置数据，减少外部 API 调用
  - 游客/免费用户强制降级为基础问答模式
  - 发送告警通知运维
- **恢复条件**：
  - 下一个自然日 00:00 自动恢复
  - 或运维手动关闭熔断

#### 4.10.7 用户等级与成本配额

| 用户等级 | 日 Token 配额 | 日行程生成次数 | 外部 API 配额 | 模型权限 |
|---|---|---|---|---|
| 游客 | 10,000 | 1 | 100 次 | 仅小模型 |
| 免费用户 | 50,000 | 3 | 500 次 | 小模型 + 有限大模型 |
| 普通会员 | 200,000 | 10 | 2,000 次 | 全部模型 |
| 高级会员 | 1,000,000 | 50 | 10,000 次 | 全部模型 + 优先队列 |
| 管理员 | 无限制 | 无限制 | 无限制 | 全部模型 + 模型版本切换 |

- 配额耗尽后：
  - Token 配额耗尽：提示用户升级或次日恢复
  - 行程生成次数耗尽：引导购买会员
  - 外部 API 配额耗尽：降级到内置数据，不影响核心功能

---

## 5. 非功能需求

### 5.1 性能目标

| 指标 | 目标 | 说明 |
|---|---|---|
| 首次响应时间 | < 100ms | 用户发送消息后首字节返回 |
| 行程草稿生成 | < 500ms（缓存命中）/ < 3s（完整生成） | 从提交到 `draft_ready` |
| 完整行程生成 | < 8s（P99）/ < 12s（P999） | 含 POI 查询 + 编排 + 润色 |
| SSE 首 Token | < 500ms（P99）/ < 2s（P999） | LLM 流式首 Token |
| 并发连接 | ≥ 1000 / 实例 | SSE/WebSocket |
| 任务吞吐 | ≥ 50 task/s / worker | 行程生成任务 |
| 压测后恢复 | < 60s | 压测停止后 CPU/内存/连接数回到基线 |

### 5.2 可用性目标

- 系统可用性 ≥ 99.5%
- 任何单一外部依赖故障时，核心功能可用（降级到内置数据）
- LLM 服务故障时，返回兜底文案并提示用户重试
- 支持滚动发布，零停机部署

### 5.3 安全目标

- 通过 JWT 鉴权，防止未授权访问
- 所有接口限流，防止 DDoS/刷接口
- 输入输出经过安全校验
- 敏感数据脱敏存储和传输

### 5.4 可观测性目标

| 维度 | 工具 | 指标 |
|---|---|---|
| Metrics | Prometheus + Grafana | QPS、P99/P999 延迟、错误率、队列长度、Token 用量、GPU/成本指标 |
| Logs | Loki / ELK | 结构化 JSON 日志，可追踪 request_id |
| Tracing | OpenTelemetry + Jaeger/Tempo | 跨服务调用链 |
| Alerting | Prometheus Alertmanager | 延迟、错误率、队列、成本、配额告警 |

#### 5.4.1 全链路 Trace 规范

##### Request ID 生成与传播

- **生成位置**：Go Gateway，在收到请求时生成全局 `request_id`（UUID v4）
- **传播方式**：
  - HTTP Header：`X-Request-ID`
  - WebSocket/SSE：握手阶段通过 query param 或 header 传递
  - Celery Task：通过 `headers` 透传
  - vLLM 调用：通过 HTTP Header 透传
- **日志绑定**：所有服务将 `request_id` 写入每条日志，便于检索

##### Span 划分

| Span 名称 | 所属服务 | 说明 |
|---|---|---|
| `gateway.receive` | Gateway | 接收请求、鉴权、限流 |
| `gateway.forward` | Gateway | 转发到 FastAPI |
| `fastapi.chat.receive` | FastAPI | 接收聊天消息 |
| `fastapi.memory.load` | FastAPI | 加载会话状态 |
| `agent.intent_recognition` | FastAPI | 意图识别 |
| `tool.poi_search` | FastAPI/Celery | POI 查询 |
| `tool.weather_query` | FastAPI/Celery | 天气查询 |
| `tool.price_query` | FastAPI/Celery | 价格查询 |
| `tool.route_calculation` | FastAPI/Celery | 路线计算 |
| `planner.draft` | Celery | 日程编排 |
| `planner.validate` | Celery | 规则校验 |
| `planner.repair` | Celery | 修复引擎 |
| `llm.vllm.call` | vLLM | 模型推理 |
| `llm.stream.token` | vLLM | 流式 Token 生成 |
| `celery.task.execute` | Celery Worker | 任务执行 |
| `redis.read` / `redis.write` | 各服务 | Redis 操作 |
| `postgres.query` | 各服务 | 数据库查询 |

##### Trace 采样策略

| 链路类型 | 采样率 | 说明 |
|---|---|---|
| 错误请求 | 100% | 必须全采样，便于排查 |
| 行程生成核心链路 | 100% | 涉及多阶段，需完整追踪 |
| 普通聊天 | 10% | 高频，可降低采样 |
| 健康检查 / metrics | 0% | 不采样 |
| 定时任务 | 1% | 批量任务按需采样 |

##### Baggage 与上下文

- 通过 OpenTelemetry Baggage 透传：
  - `user_id`
  - `conversation_id`
  - `model_name`
  - `job_id`
- 用于日志关联、计费统计、问题定位

##### Trace 与告警联动

- 当某个 Span 的 P99 延迟超过阈值时，自动关联 Trace 样本
- 告警信息中包含 deep link，点击直达 Jaeger 对应 Trace

---

## 6. 数据模型

### 6.1 用户表 `users`

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE,
    phone VARCHAR(20),
    password_hash VARCHAR(255),
    role VARCHAR(20) DEFAULT 'user',  -- guest / user / admin
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 6.2 用户资料表 `user_profiles`

```sql
CREATE TABLE user_profiles (
    user_id UUID PRIMARY KEY REFERENCES users(id),
    personal JSONB,  -- 饮食偏好、兴趣、节奏等
    preferences JSONB,  -- 系统级偏好
    frequent_destinations JSONB,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 6.3 会话表 `conversations`

```sql
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    title VARCHAR(255),
    status VARCHAR(20) DEFAULT 'active',  -- active / archived / deleted
    state_snapshot JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    archived_at TIMESTAMPTZ
);
```

### 6.4 消息表 `messages`

```sql
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id),
    role VARCHAR(20),  -- user / assistant / system
    content TEXT,
    token_count INT DEFAULT 0,  -- 单条消息 Token 数
    metadata JSONB,  -- token 用量、模型、意图等
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_messages_conversation_created ON messages(conversation_id, created_at);
```

- `token_count` 直接存储单条消息 Token 用量，减少实时统计成本
- `metadata` 中保留 `input_tokens` / `output_tokens` / `model_name` 等细节

### 6.5 规划任务表 `planning_jobs`

```sql
CREATE TABLE planning_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id),
    user_id UUID REFERENCES users(id),
    queue_name VARCHAR(50) DEFAULT 'planning',  -- planning / poi / memory / default
    status VARCHAR(20),  -- pending / running / completed / failed / cancelled
    input_requirements JSONB,
    result JSONB,
    token_usage JSONB,
    latency_ms INT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_planning_jobs_queue_status ON planning_jobs(queue_name, status);
CREATE INDEX idx_planning_jobs_user_created ON planning_jobs(user_id, created_at);
```

- `queue_name` 用于记录任务所属队列，便于问题定位、死信分类、队列级监控

### 6.6 任务事件表 `planning_job_events`

```sql
CREATE TABLE planning_job_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID REFERENCES planning_jobs(id),
    stage VARCHAR(50),
    status VARCHAR(20),  -- running / completed / failed
    payload JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 6.7 行程表 `itineraries`

```sql
CREATE TABLE itineraries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID REFERENCES planning_jobs(id),
    user_id UUID REFERENCES users(id),
    conversation_id UUID REFERENCES conversations(id),
    destination VARCHAR(100),
    days INT,
    content JSONB,
    proposal_text TEXT,
    is_favorite BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 6.8 JSONB Schema 校验

- 所有 JSONB 字段通过应用层 Pydantic 模型写入前校验
- 可选在 PostgreSQL 中启用 `jsonb_schema_validation` 插件或触发器做二次校验
- 关键 JSONB 字段的 Schema：
  - `users.preferences`
  - `conversations.state_snapshot`
  - `planning_jobs.input_requirements`
  - `planning_jobs.result`
  - `planning_jobs.token_usage`
  - `messages.metadata`
  - `itineraries.content`
- 校验失败时记录错误日志并拒绝写入，防止脏数据

---

## 7. API 设计

### 7.1 认证相关

#### POST /api/v1/auth/register
注册新用户。

**请求：**
```json
{
  "email": "user@example.com",
  "password": "string"
}
```

**响应（201 Created）：**
```json
{
  "user_id": "uuid",
  "access_token": "jwt",
  "refresh_token": "jwt",
  "token_type": "bearer"
}
```

**错误响应：**
- `400 Bad Request`：参数错误或邮箱已注册
- `429 Too Many Requests`：IP 注册频率超限

---

##### 通用 HTTP 状态码

| 状态码 | 场景 | 说明 |
|---|---|---|
| 200 OK | 请求成功 | 通用成功 |
| 201 Created | 创建成功 | 注册、创建会话、提交任务 |
| 204 No Content | 操作成功无返回体 | 删除、取消 |
| 400 Bad Request | 请求参数错误 | 缺少必填字段、格式错误 |
| 401 Unauthorized | 未授权 | Token 缺失/过期/无效/吊销 |
| 403 Forbidden | 禁止访问 | 权限不足、设备不一致 |
| 404 Not Found | 资源不存在 | 会话/任务/行程不存在 |
| 409 Conflict | 资源冲突 | 重复提交、状态冲突 |
| 429 Too Many Requests | 限流 | 超过 QPS / Token 配额 |
| 503 Service Unavailable | 服务不可用 | 后端熔断、模型过载 |
| 504 Gateway Timeout | 网关超时 | 上游服务超时 |

#### POST /api/v1/auth/login
登录。

#### POST /api/v1/auth/refresh
刷新 access token。

#### POST /api/v1/auth/guest
获取游客 token。

**响应：**
```json
{
  "access_token": "jwt",
  "token_type": "bearer",
  "expires_in": 86400
}
```

### 7.2 会话相关

#### POST /api/v1/conversations
创建新会话。

**响应（201 Created）：**
```json
{
  "conversation_id": "uuid",
  "session_id": "uuid",
  "created_at": "2026-06-15T00:00:00Z"
}
```

#### GET /api/v1/conversations
列出用户会话列表。

**查询参数：**
- `page`：页码，默认 1
- `page_size`：每页数量，默认 20，最大 100
- `status`：筛选状态 `active / archived / deleted`

**响应（200 OK）：**
```json
{
  "total": 100,
  "page": 1,
  "page_size": 20,
  "items": [
    {
      "conversation_id": "uuid",
      "title": "杭州3日游",
      "status": "active",
      "updated_at": "2026-06-15T00:00:00Z"
    }
  ]
}
```

#### GET /api/v1/conversations/{id}
获取会话详情和状态。

#### POST /api/v1/conversations/batch/archive
批量归档会话。

**请求：**
```json
{
  "conversation_ids": ["uuid1", "uuid2", "uuid3"]
}
```

**响应（200 OK）：**
```json
{
  "success_count": 3,
  "failed_count": 0,
  "failed_ids": []
}
```

#### POST /api/v1/conversations/batch/delete
批量删除会话。

#### POST /api/v1/itineraries/batch/favorite
批量收藏行程。

#### POST /api/v1/itineraries/batch/export
批量导出行程（PDF/JSON）。

### 7.3 实时对话

#### GET /api/v1/chat/stream?conversation_id=xxx&timeout=1800
SSE 流式对话入口。

**请求头：**
- `Authorization: Bearer <token>`
- `X-Request-ID: <request_id>`（可选，Gateway 未提供时自动生成）

**查询参数：**
- `conversation_id`：会话 ID（必填）
- `timeout`：连接超时时间，单位秒，默认 1800（30min），最大 3600（1h）
- `last_event_id`：断线重连时传入，服务端从该事件 ID 后继续推送

> 所有 API 响应均返回 `X-Request-ID`，前端可记录用于问题排查。

**发送消息：** 通过 POST `/api/v1/chat/message` 提交用户消息，然后由 SSE 推送结果。

**POST /api/v1/chat/message**

```json
{
  "conversation_id": "uuid",
  "content": "我想去杭州玩3天",
  "stream": true
}
```

**SSE 事件示例：**

```
event: stage
data: {"stage":"intent_ready","status":"running","job_id":"uuid"}

event: token
data: {"chunk":"第","job_id":"uuid"}

event: token
data: {"chunk":"一天","job_id":"uuid"}

event: stage
data: {"stage":"completed","status":"completed","job_id":"uuid"}

event: done
data: {}
```

### 7.4 行程相关

#### POST /api/v1/itineraries
提交行程生成任务。

```json
{
  "conversation_id": "uuid",
  "requirements": {
    "destination": "杭州",
    "days": 3,
    "budget_range": "medium",
    "preferences": ["美食", "西湖"]
  }
}
```

**响应：**
```json
{
  "job_id": "uuid",
  "status": "pending",
  "stream_url": "/api/v1/itineraries/uuid/stream"
}
```

#### GET /api/v1/itineraries/{id}/stream
SSE 推送行程生成进度和结果。

#### GET /api/v1/itineraries/{id}
获取行程结果。

#### GET /api/v1/itineraries
获取用户历史行程列表。

#### POST /api/v1/itineraries/{id}/favorite
收藏行程。

### 7.5 任务相关

#### GET /api/v1/jobs/{id}
查询任务状态。

#### POST /api/v1/jobs/{id}/cancel
取消任务。

### 7.6 系统相关

#### GET /api/v1/health
健康检查。

#### GET /api/v1/metrics
Prometheus metrics（Gateway 暴露）。

---

## 8. 模型层详细设计

### 8.1 vLLM 部署

#### 8.1.1 基础模型选择

| 场景 | 推荐模型 | 显存需求 |
|---|---|---|
| 开发/测试 | Qwen2.5-7B-Instruct | 16GB |
| 生产 | Qwen2.5-14B-Instruct 或 DeepSeek-V2.5 | 32–48GB |
| 高性能 | DeepSeek-V3 / Qwen2.5-72B | 多卡 |

#### 8.1.2 vLLM 启动参数

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-14B-Instruct \
  --enable-lora \
  --lora-modules travel-chat=./adapters/travel-chat-v1 \
                 travel-plan=./adapters/travel-plan-v1 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --dtype half
```

#### 8.1.3 vLLM 集群扩缩容与稳定性

##### 扩缩容触发条件

| 指标 | 扩容阈值 | 缩容阈值 | 冷却时间 |
|---|---|---|---|
| GPU 利用率 | > 80% 持续 60s | < 30% 持续 300s | 180s |
| 请求队列长度 | > 20 持续 60s | < 5 持续 300s | 180s |
| P95 TTFT | > 2s 持续 120s | < 500ms 持续 300s | 180s |
| Batch 等待时间 | > 500ms 持续 60s | - | 180s |

- **扩容步长**：每次 +1 Pod，最大副本数 5
- **缩容步长**：每次 -1 Pod，最小副本数 1
- **缩容保护**：仅缩容无活跃请求的 Pod，避免中断正在进行的推理

##### 模型加载与预热

- 新 Pod 启动后：
  1. 加载基础模型（约 30–60s，取决于模型大小和存储）
  2. **预加载 Top 3 常用 LoRA adapter**：
     - `travel-chat-v1`（简单问答/意图识别）
     - `travel-plan-v1`（行程生成/润色）
     - `travel-repair-v1`（异常解释）
  3. 健康检查通过后再加入 Service Endpoint
- 使用 Readiness Probe 探测 `/health`，Timeout 5s，失败 3 次不加入流量
- **LoRA 按需加载风险**：非常用 adapter 首次请求时延迟可能升高 500ms–2s
  - 缓解：核心 adapter 常驻内存，非常用 adapter 在低谷期预加载
  - 监控：记录 adapter 加载耗时，> 1s 触发告警

##### LoRA Adapter 热加载

- vLLM 启动时通过 `--lora-modules` 预注册常用 adapter
- 新增 adapter 时：
  1. 将 adapter 文件推送到共享存储（NFS / S3 / PVC）
  2. 调用 vLLM `/v1/load_lora_adapter` 接口动态加载
  3. 无需重启 vLLM 服务
- adapter 卸载：
  - 调用 `/v1/unload_lora_adapter`
  - 或使用滚动升级：新 Pod 带新 adapter，老 Pod 逐步下线

##### 任务迁移与优雅缩容

- vLLM Pod 缩容前：
  1. 将 Pod 标记为 `draining`，不再接收新请求
  2. 等待已有请求完成（最大等待 60s）
  3. 超时后强制终止，由客户端重试机制兜底
- FastAPI 端需实现请求重试：
  - vLLM 返回 503 或连接失败时，重试到其他 Pod
  - 最大重试 3 次，总超时 30s

##### 多副本路由策略

- Gateway 到 vLLM 使用 K8s Service 负载均衡
- 可选 Session Affinity：同一用户的连续请求尽量路由到同一 Pod，提高 cache hit
- 长文本请求（> 4k tokens）优先路由到 GPU 显存更充足的 Pod

#### 8.1.4 模型版本管理

- MLflow 记录每个 LoRA 版本的：
  - 训练数据版本
  - 超参数
  - 评估指标（BLEU、ROUGE、人工评分）
  - Artifact 路径
- vLLM 通过 `model` 参数动态切换 adapter：
  - `travel-chat-v1`
  - `travel-plan-v1`

### 8.2 LoRA 微调流水线

#### 8.2.1 数据准备

```python
# ml/training/prepare_data.py
# 输入：raw_dialogues.jsonl
# 输出：train.jsonl / val.jsonl
```

数据格式：
```json
{
  "messages": [
    {"role": "system", "content": "你是旅行规划助手..."},
    {"role": "user", "content": "我想去杭州玩3天"},
    {"role": "assistant", "content": "好的，我建议您..."}
  ]
}
```

#### 8.2.2 训练脚本

```python
# ml/training/train_lora.py
# 使用 PEFT + Transformers + TRL
# 配置：rank=64, alpha=128, lr=2e-4, batch_size=4
```

#### 8.2.3 评估

```python
# ml/training/evaluate.py
# 指标：格式正确率、事实一致性、用户满意度
```

#### 8.2.4 部署

```python
# ml/training/merge_and_upload.py
# 合并 LoRA → 推送到模型仓库 → 注册 MLflow
```

### 8.3 LLM 客户端设计

`core/llm_client.py` 改造后接口：

```python
class LLMClient:
    async def chat(
        self,
        messages: list,
        model: str = "default",
        temperature: float = 0.7,
        stream: bool = False,
    ) -> str | AsyncIterator[str]:
        ...

    async def structured_call(
        self,
        messages: list,
        output_schema: Type[T],
        model: str = "default",
    ) -> T:
        ...

    async def stream_chat(
        self,
        messages: list,
        model: str = "default",
    ) -> AsyncIterator[str]:
        ...
```

### 8.4 模型路由

`core/model_router.py`：

```python
class ModelRouter:
    def select_model(self, task: TaskType, context: Context) -> str:
        if task == TaskType.INTENT or task == TaskType.FAQ:
            return "travel-chat-v1"  # 7B
        elif task == TaskType.ITINERARY or task == TaskType.POLISH:
            return "travel-plan-v1"  # 14B
        else:
            return "default"
```

---

## 9. Agent 工作流详细设计

### 9.1 意图识别

- 输入：用户消息 + 当前会话状态
- 输出：`IntentResult`（意图类型 + 提取的实体 + 缺失字段）
- 模型：小模型（travel-chat LoRA）
- 优化：常见意图加规则 fast path，不走模型

### 9.2 信息收集

- 并行触发：
  - POI 查询（city + 偏好关键词）
  - 天气查询（city + dates）
  - 价格查询（POI 列表）
  - 路线距离矩阵（POI 两两之间）
- 每个子任务 3s 超时，结果合并后进入编排

### 9.3 日程编排

- 使用现有 `daily_scheduler.py` 作为 baseline
- 输入：POI 列表、天数、偏好、约束
- 输出：每日活动安排（含时间、地点、交通）

### 9.4 规则校验与修复

- 使用现有 `rule_validator.py` + `repair.py`
- 新增跨城市、节假日、路线冲突场景
- 修复失败时转用户澄清

### 9.5 文案润色

- 输入：结构化行程
- 输出：自然语言行程描述
- 模型：大模型（travel-plan LoRA）
- 流式输出到前端

---

## 10. 工具层详细设计

### 10.1 工具基类

```python
class Tool(ABC):
    name: str
    timeout: float = 3.0
    retries: int = 3
    cache_ttl: int = 3600

    @abstractmethod
    async def execute(self, params: dict) -> ToolResult:
        pass

    async def run(self, params: dict) -> ToolResult:
        # 1. 查缓存
        # 2. 执行（含重试、超时）
        # 3. 降级
        # 4. 写缓存
        pass

class ToolResult(BaseModel):
    data: Any
    data_source: Literal["api", "built_in", "fallback", "unavailable"]
    confidence: float = 1.0  # 0.0 - 1.0
    is_fallback: bool = False
    fallback_reason: str | None = None
    latency_ms: int
    retries: int = 0
```

### 10.2 工具列表

| 工具 | 类 | 数据源 | 缓存 |
|---|---|---|---|
| POI 搜索 | `POISearchTool` | 内置 + 高德 + Tavily | 6h |
| 天气查询 | `WeatherTool` | 和风天气 | 1h |
| 价格查询 | `PriceTool` | OTA/门票平台 | 30min |
| 路线计算 | `RouteTool` | 高德 Distance Matrix | 24h |
| 地图搜索 | `MapSearchTool` | 高德/百度 | 6h |

### 10.3 熔断与降级

- 使用 `pybreaker` 或自研熔断器
- 失败率阈值：50%
- 恢复时间：30s
- 降级顺序：真实 API → 内置数据 → 占位提示

---

## 11. 前端改造

### 11.1 新增/改造文件

```
frontend/src/
  hooks/
    useSSE.ts              # 新增：SSE 连接管理
    useChat.ts             # 改造：统一聊天逻辑
  components/
    ChatPanel.tsx          # 改造：接入 SSE + 流式渲染
    MessageBubble.tsx      # 改造：支持 markdown + 流式文字
    StreamingText.tsx      # 新增：逐字渲染组件
    ItineraryPanel.tsx     # 改造：实时展示阶段进度
  app/
    page.tsx               # 改造：移除 WebSocket 初始化
  stores/
    chatStore.ts           # 改造：增加 stream state
```

### 11.2 SSE Hook 设计

```typescript
interface UseSSEOptions {
  conversationId: string;
  token: string;
  onMessage: (msg: SSEMessage) => void;
  onError: (error: Error) => void;
}

function useSSE(options: UseSEOptions): {
  isConnected: boolean;
  reconnect: () => void;
  close: () => void;
};
```

### 11.3 流式渲染

- LLM Token 通过 SSE `token` 事件到达
- `StreamingText` 组件逐字追加到消息内容
- 支持暂停/继续、复制完整内容

### 11.4 阶段展示

- 顶部进度条展示当前阶段
- 每个阶段配图标和说明
- 失败时展示重试按钮

---

## 12. 部署与运维

### 12.1 K8s 部署清单

```
k8s/
  namespace.yaml
  configmap.yaml
  secret.yaml
  gateway-deployment.yaml
  gateway-service.yaml
  gateway-ingress.yaml
  backend-deployment.yaml
  backend-service.yaml
  celery-worker-deployment.yaml
  celery-beat-deployment.yaml
  vllm-deployment.yaml
  vllm-service.yaml
  redis-statefulset.yaml
  postgres-statefulset.yaml
  hpa.yaml
  pdb.yaml                 # Pod Disruption Budget
  backup-cronjob.yaml      # 数据备份任务
```

### 12.2 资源规格

#### 服务资源请求与限制

| 服务 | 副本数 | CPU 请求 | CPU 限制 | 内存请求 | 内存限制 | GPU | 说明 |
|---|---|---|---|---|---|---|---|
| Go Gateway | 3 | 1 | 2 | 512Mi | 2Gi | - | 高并发网络 I/O，CPU 密集型 |
| FastAPI | 3 | 1 | 3 | 1Gi | 2Gi | - | 业务逻辑 + SSE 连接保持 |
| Celery Worker（default） | 3 | 1 | 2 | 1Gi | 2Gi | - | 普通异步任务 |
| Celery Worker（planning） | 5 | 2 | 4 | 2Gi | 4Gi | - | 重计算任务 |
| Celery Worker（poi） | 3 | 1 | 3 | 1Gi | 2Gi | - | 外部 API 调用 |
| Celery Beat | 1 | 0.5 | 1 | 512Mi | 1Gi | - | 定时任务调度 |
| vLLM（7B 模型） | 1–3 | 4 | 8 | 16Gi | 32Gi | 1×A10 (24GB) | 中小模型推理 |
| vLLM（14B 模型） | 1–5 | 8 | 16 | 32Gi | 64Gi | 1×A100 (40/80GB) | 大模型推理 |
| Redis Cluster | 6（3主3从） | 1 | 2 | 4Gi | 8Gi | - | 缓存/状态/队列 |
| PostgreSQL | 2（主从） | 2 | 4 | 4Gi | 8Gi | - | 主库 + 热备只读从库 |
| Next.js Frontend | 2 | 0.5 | 1 | 512Mi | 1Gi | - | SSR 渲染 |

#### 生产环境推荐机型

| 服务 | 推荐节点 | 说明 |
|---|---|---|
| Go Gateway | 4C8G × 3 | 网络 I/O 型 |
| FastAPI | 4C8G × 3 | 通用计算型 |
| Celery Worker | 8C16G × 5（planning） | 计算密集型 |
| vLLM（14B） | 8C64G + 1×A100 × 2 | GPU 推理型 |
| Redis | 4C16G × 6 | 内存型 |
| PostgreSQL | 4C16G × 2 | IO 优化型 SSD |

#### 资源调度策略

- **QoS 等级**：
  - Gateway / FastAPI / vLLM：Guaranteed（requests = limits）
  - Celery Worker：Burstable
  - Beat / Frontend：Burstable
- **节点亲和性**：
  - vLLM Pod 调度到 GPU 节点（`nodeSelector: accelerator: nvidia`）
  - Redis / PostgreSQL 调度到高 IO 节点（SSD）
- **反亲和性**：
  - 同服务副本尽量分布在不同节点
  - Redis 主从不在同一节点，避免节点故障同时丢失主从
- **Pod Disruption Budget**：
  - Gateway：`minAvailable: 2`
  - FastAPI：`minAvailable: 2`
  - Redis：每个主节点至少 1 个副本可用

### 12.3 HPA 配置

| 服务 | 最小副本 | 最大副本 | 扩容指标 |
|---|---|---|---|
| Gateway | 2 | 10 | CPU > 70% |
| FastAPI | 3 | 20 | CPU > 70% / 请求队列 |
| Celery Worker（planning） | 2 | 30 | 队列长度 > 50 |
| Celery Worker（poi） | 2 | 20 | 队列长度 > 30 |
| vLLM | 1 | 5 | GPU 利用率 > 80% 持续 60s / 队列长度 > 20 持续 60s |

**vLLM 扩缩容补充说明**：
- 扩容优先级：队列长度 > GPU 利用率 > TTFT
- 缩容必须等待 Pod 上无活跃推理请求（draining 模式）
- LoRA adapter 通过共享存储 + 热加载机制更新，不中断推理服务

### 12.4 发布策略与回滚

#### 灰度发布

- **发布流程**：
  1. 构建新镜像并推送到镜像仓库
  2. Argo Rollouts 创建新 ReplicaSet，初始副本数 = 总数 10%
  3. 观察 5 分钟，监控：错误率、P99 延迟、业务核心指标
  4. 无异常则逐步扩容：25% → 50% → 75% → 100%
  5. 每阶段观察 3 分钟

- **自动回滚触发条件**：
  | 指标 | 阈值 | 触发动作 |
  |---|---|---|
  | 错误率 | > 5% 持续 2min | 自动回滚 |
  | P99 延迟 | > 基线 200% 持续 3min | 自动回滚 |
  | 5xx 错误数 | > 100 / min | 自动回滚 |
  | Pod 重启次数 | > 3 次 / 5min | 暂停发布，人工确认 |
  | 核心业务失败率 | 行程生成失败率 > 10% | 立即回滚 |

- **回滚方式**：
  - 自动：Argo Rollouts 自动将流量切回旧版本
  - 手动：`kubectl argo rollouts abort <rollout>`
  - 数据库变更回滚：通过 Alembic down-grade 脚本

#### 数据库变更回滚

- 每个 migration 必须包含 `upgrade()` 和 `downgrade()`
- 破坏性变更（如删除列）需分两步：
  1. 先发布兼容新旧 schema 的代码
  2. 再发布只使用新 schema 的代码
- 发布前在 staging 环境验证 downgrade

### 12.5 灾备与数据备份

#### 数据备份策略

| 数据 | 备份方式 | 频率 | 保留周期 | 存储位置 |
|---|---|---|---|---|
| PostgreSQL 全量 | pg_basebackup | 每日 02:00 | 7 天 | 对象存储（S3/OSS） |
| PostgreSQL WAL | 实时归档 | 持续 | 30 天 | 对象存储 |
| Redis RDB | BGSAVE | 每日 03:00 | 7 天 | 对象存储 |
| Redis AOF | 实时追加 | 持续 | 7 天 | 本地 + 对象存储 |
| MLflow 模型 | 完整复制 | 每次发布 | 永久 | 对象存储 + 异地 |
| 配置 / Secret | GitOps 仓库 | 每次变更 | 永久 | Git + 加密对象存储 |

#### 恢复目标

| 数据 | RPO | RTO |
|---|---|---|
| PostgreSQL | < 5 分钟（WAL 归档） | < 30 分钟 |
| Redis | < 1 小时（RDB + AOF） | < 15 分钟 |
| 会话状态 | < 1 小时 | < 15 分钟 |
| 模型文件 | 0（对象存储多副本） | < 10 分钟 |

#### 多可用区部署

- **部署拓扑**：
  - 可用区 A（AZ-A）+ 可用区 B（AZ-B）
  - Gateway / FastAPI / Celery 副本跨 AZ 分布
  - Redis Cluster 主从跨 AZ 分布
  - PostgreSQL 主从跨 AZ 部署（主在 AZ-A，热备在 AZ-B）
  - vLLM 部署在单个 AZ（GPU 节点池通常集中），但模型文件异地备份

- **容灾切换**：
  - Redis：单主节点故障时自动 failover，跨 AZ 不影响
  - PostgreSQL：主库故障时手动/自动提升从库为主库
  - Gateway / FastAPI：K8s 自动将故障 AZ 流量切到健康 AZ
  - vLLM：单 AZ 故障时从模型文件恢复，RTO < 10 分钟

- **Pod Disruption Budget**：
  - 确保升级或节点维护时，每个 AZ 至少保留必要副本

#### 灾备演练

- **频率**：每月一次数据恢复演练
- **演练内容**：
  - PostgreSQL 全量备份 + WAL 恢复到临时实例
  - Redis RDB + AOF 恢复到临时集群
  - vLLM 模型文件从对象存储恢复到新 Pod
- **演练目标**：
  - 验证 RPO/RTO 是否达标
  - 更新灾备操作手册
  - 发现备份链完整性问题

### 12.6 CI/CD

- GitHub Actions：
  - 代码提交后运行 lint、test、build
  - 合并到 main 后构建镜像并推送
  - ArgoCD / kubectl 自动同步到 K8s

### 12.7 可观测性

- Prometheus 抓取：Gateway、FastAPI、Celery、vLLM
- Grafana 大盘：
  - 总 QPS / 错误率 / P99 / P999 延迟
  - 各阶段任务耗时
  - Token 用量与成本
  - 缓存命中率
  - 队列长度与消费速率
  - GPU 利用率 / 显存 / vLLM batch 大小
  - 用户等级配额消耗
- Loki 收集结构化日志

---

## 13. 测试策略

### 13.1 单元测试

- 覆盖：模型路由、工具基类、修复引擎、规则校验、状态合并
- 要求：核心模块覆盖率 ≥ 90%

### 13.2 集成测试

- 覆盖：API 端到端、SSE 流式、Celery 任务、Redis 状态一致性
- 使用 testcontainers 启动 PG + Redis

### 13.3 性能测试

- 使用 locust/k6 模拟并发用户
- 目标：1000 并发 SSE 连接稳定 5 分钟
- 监控：内存、CPU、连接数、P99 延迟

### 13.4 混沌测试

- 模拟 Redis 故障、LLM 超时、外部 API 故障
- 验证降级策略和系统可用性

### 13.5 安全测试

- 提示注入测试集
- 越权访问测试
- 敏感数据泄露扫描

---

## 14. 里程碑与排期

### 14.1 里程碑

| 阶段 | 时间 | 交付物 | 验收标准 |
|---|---|---|---|
| M1 基座 | 第 1–2 周 | Go 网关 + JWT + 限流 + SSE 骨架 | 网关可转发、SSE 可连、限流生效 |
| M2 异步化 | 第 3–4 周 | Celery + Redis 缓存 + 任务状态机 | 行程生成走异步、缓存命中 |
| M3 Agent 升级 | 第 5–6 周 | 记忆归档/清理 + 修复引擎增强 + 工具降级 | 高并发记忆一致、异常可自愈 |
| M4 模型层 | 第 7–9 周 | vLLM 部署 + LoRA 训练 + 大小模型路由 | 本地可推理、LoRA 效果达标 |
| M5 工程化 | 第 10–11 周 | K8s + 监控 + MLflow + 安全 | 可水平扩展、可观测、安全测试通过 |
| M6 优化 | 第 12 周 | 成本优化 + 性能调优 + 文档 | 性能目标达成、成本降低 |

### 14.2 关键路径

```
M1 网关/SSE → M2 Celery/缓存 → M3 Agent/记忆 → M4 vLLM/LoRA → M5 K8s → M6 优化
```

---

## 15. 风险与假设

### 15.1 风险

| 风险 | 影响 | 缓解措施 |
|---|---|---|
| LoRA 训练数据不足 | 模型效果差 | 先用 prompt engineering + RAG，同步积累数据 |
| GPU 资源不足 | vLLM 无法部署 | 先用云端兼容 API mock，后续迁移 |
| 外部 API 配额限制 | POI/天气查询失败 | 强化内置 fallback + 缓存 |
| 高并发下 Redis 瓶颈 | 延迟升高 | Redis Cluster + 连接池优化 |
| 改造成期超出预期 | 无法按时交付 | 分阶段 MVP，每阶段可独立演示 |
| vLLM LoRA 按需加载延迟 | 首请求 TTFT 升高 | 核心 LoRA 预加载，非常用 LoRA 低谷期预热 |
| Redis 乐观锁写冲突 | 会话状态覆盖/丢失 | 核心会话写操作使用 Redlock 分布式锁 |
| 成本失控 | 免费用户耗尽配额后仍调用大模型 | 用户等级配额 + 成本熔断开关 + 强制降级 |

### 15.2 假设

- 开发/测试环境可访问 GPU 或云端 LLM API
- 可获取或合成至少 5000 条旅行对话数据用于 LoRA
- 用户接受 SSE 替代 WebSocket 的实时交互方式
- 团队熟悉 Kubernetes 和 Prometheus/Grafana

---

## 16. 附录

### 16.1 环境变量清单

```bash
# Gateway
GATEWAY_PORT=8080
GATEWAY_WORKERS=4
JWT_SECRET=xxx
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Backend
APP_HOST=0.0.0.0
APP_PORT=8000
DATABASE_URL=postgresql+asyncpg://...
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=xxx

# LLM / vLLM
VLLM_BASE_URL=http://vllm:8000/v1
VLLM_API_KEY=not-needed
DEFAULT_MODEL=travel-plan-v1
SMALL_MODEL=travel-chat-v1

# External APIs
AMAP_KEY=xxx
WEATHER_KEY=xxx
TAVILY_API_KEY=xxx

# Celery
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2

# Monitoring
PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus
```

### 16.2 关键指标定义

| 指标 | 计算方式 |
|---|---|
| QPS | 每秒成功请求数 |
| P99 延迟 | 99% 请求响应时间 |
| TTFT | 从发送请求到收到首个 Token 的时间 |
| 缓存命中率 | 命中缓存请求数 / 总请求数 |
| 任务成功率 | 成功任务数 / 总任务数 |
| Token 成本 | 每千次请求平均 Token 花费 |

---

## 17. 评审记录

| 日期 | 评审人 | 结论 | 备注 |
|---|---|---|---|
| 2026-06-15 | — | 初稿 | 待评审 |
| 2026-06-15 | AI 评审 | 修订中 | 补充架构层、功能层、非功能层、部署运维层细节 |
| 2026-06-15 | AI 评审 | 修订完成 | 补充关键遗漏点：资源规格、配置中心、数据模型、API 示例、安全加密、风险点 |

---

**下一步建议**：进入 M1 开发，优先实现 Go 网关 + SSE 骨架 + Redis 缓存层。