# 旅游Agent2 技术蓝图 v3.0 上半部分：用户理解与画像召回
> 覆盖第0-4层 + 多Agent架构 + 编排调度 + 感知与意图 + 用户画像
> 版本日期：2026-06-18

---

## 文档定位

上半部分聚焦 **"用户进来 → 系统理解用户 → 召回画像 → 输出结构化的旅行需求"** 这一链路，包含：

| 章节 | 层级 | 核心问题 | 产出物 |
|------|------|---------|--------|
| 第0章 | 总览 | 整体架构长什么样？ | 8层架构图 + 技术选型表 |
| 第0.5章 | 多Agent架构 | 哪些Agent参与？各自做什么？ | 6核心+3增值Agent定义 |
| 第1章 | 编排调度层 | 如何调度这些Agent？出错了怎么办？ | LangGraph完整实现 |
| 第2章 | 感知层 | 用户说什么/传什么？系统怎么接收？ | 4类输入处理方案 |
| 第3章 | 意图理解层 | 用户想要什么？Constraints有哪些？ | 槽位抽取+情感+可行性校验 |
| 第4章 | 用户画像层 | 这个用户是谁？之前来过吗？ | 画像召回+记忆读写+隐私 |

**下半部分**（知识检索与行程规划）覆盖第5-8层，独立成文档。

---

## 0. 总览：8层架构全景图

```
用户输入
  │
  ▼
┌──────────────────────────────────────────────────────┐
│ 第2层 感知层        │ 文本SSE + 图片(PaddleOCR) + PDF + Webhook │
├──────────────────────────────────────────────────────┤
│ 第3层 意图理解层    │ 槽位抽取 + 情感识别 + 歧义消解 + 可行性校验 │  ← 上半部分
├──────────────────────────────────────────────────────┤
│ 第4层 用户画像层    │ 短期Redis上下文 + 长期pgvector记忆         │  ← 上半部分
├──────────────────────────────────────────────────────┤
│ 第5层 知识库层      │ 结构化DB + RAG混合检索 + 实时API          │  ← 下半部分
├──────────────────────────────────────────────────────┤
│ 第6层 规划决策引擎  │ OR-Tools CP-SAT v4.0 + 场景参数 + 三模式重规划 │ ← 下半部分
├──────────────────────────────────────────────────────┤
│ 第7层 工具调用层    │ 11类Function Calling工具                 │  ← 下半部分
├──────────────────────────────────────────────────────┤
│ 第8层 交互输出层    │ Markdown + 地图 + PDF + Excel + 语音     │  ← 下半部分
└──────────────────────────────────────────────────────┘
  ▲
第1层 编排调度层（LangGraph StateGraph 贯穿全链路）
```

**核心设计哲学**：上半部分解决 **"理解用户是谁、用户想要什么"** 的问题，下半部分解决 **"基于用户需求找到合适的景点并规划最优路线"** 的问题。两层之间通过 `AgentState.poi_candidates`（Top-15结构化POI列表）和 `AgentState.slots`（完整TravelSlots）衔接。

---

### 全栈技术选型

| 层级 | 技术选型 | 选型理由 | 降级方案 | 负责人 |
|------|---------|---------|---------|--------|
| **编排框架** | LangGraph StateGraph + PostgreSQL Checkpoint | 循环/分支/断点/持久化 | 手动FastAPI流水线 | 架构组 |
| **LLM(复杂)** | Qwen2.5-72B-Instruct + 旅游LoRA | 中文最优、vLLM私有化 | TGI 14B-AWQ / llama.cpp CPU | 算法组 |
| **LLM(轻量)** | Qwen2.5-7B-AWQ | 意图/情感/槽位 | 云端API | 算法组 |
| **多模态(MVP)** | PaddleOCR + PyMuPDF | MVP不调用VL，仅提取文字 | 提示"功能开发中" | 前端组 |
| **向量库** | pgvector (PostgreSQL 16) | 结构化+向量统一 | Redis预计算→本地LRU | 后端组 |
| **缓存** | Redis 7 | 会话/限流/求解缓存 | 本地LRU | 后端组 |
| **推理引擎** | vLLM | continuous batching, LoRA热加载 | TGI / llama.cpp | 架构组 |
| **Embedding** | bge-large-zh-v1.5 (768维) | 中文MTEB最优 | text2vec | 算法组 |
| **前端** | Next.js 15 + TypeScript + Tailwind | 现有技术栈 | — | 前端组 |
| **后端** | FastAPI + Celery | 现有技术栈 | — | 后端组 |
| **部署** | Docker + K8s | 现有技术栈 | Docker单容器 | 运维组 |

---

## 0.5 多Agent架构（6核心 + 3增值）

### 0.5.1 架构设计原则

1. **单一职责**：每个Agent只做一件事，通过"绝对不能做"清单约束
2. **数据契约**：Agent之间通过 TypedDict 传递状态，不直接调用彼此的内部方法
3. **故障隔离**：单个Agent失败不影响整个链路，通过降级策略保证可用性
4. **可观测性**：每个Agent的输入/输出/耗时/错误都记录到 execution_trace

### 0.5.2 6个核心Agent

| # | Agent名称 | 所在LangGraph Node | 核心职责 | 绝对不能做 |
|---|----------|-------------------|---------|-----------|
| ① | **DemandParserAgent** | `understand_node` | 从用户输入抽取TravelSlots + 情感 + 可行性校验 | 查数据库；调用求解器；修改画像 |
| ② | **UserProfileRecallAgent** | `profile_node` | 召回用户画像（短期Redis + 长期pgvector） | 修改画像（只读）；调用LLM |
| ③ | **TravelRetrievalRAGAgent** | `retrieve_node` | 混合检索Top-15结构化POI | 编造POI；LLM生成虚假数据 |
| ④ | **ItineraryPlannerAgent** | `plan_node` | 调用CP-SAT v4.0求解最优行程 | 用LLM规划路线；编造时间/费用 |
| ⑤ | **FactCheckAgent** | `factcheck_node` | 回查PostgreSQL校验行程真实性 | 修改行程；触发重规划 |
| ⑥ | **OutputFormatAgent** | `output_node` | LLM润色文案 + 格式化输出 | 修改行程内容；编造事实 |

### 0.5.3 3个可选增值Agent

| # | Agent名称 | 触发阶段 | 前置依赖 |
|---|----------|---------|---------|
| ⑦ | **BookingToolAgent** | P2（体验升级期） | 外部API接入框架 + 商务合作（携程/12306/飞猪） |
| ⑧ | **EmergencyAssistantAgent** | P1（核心增强期） | 实时天气API + 交通API + 用户推送渠道 |
| ⑨ | **MultiPersonSyncAgent** | P3（规模化期） | 多人会话管理 + 预算分摊算法 + 投票决策 |

### 0.5.4 Agent间数据契约

```python
# ① → ② 的契约：TravelSlots + feasibility_report
slots_output = {
    "slots": TravelSlots,           # 完整槽位
    "feasibility_report": {         # 可行性校验结果
        "feasible": bool,
        "conflicts": [{"type": str, "level": "error|warning|info",
                        "message": str, "suggestions": [str]}]
    },
    "intent": "new_itinerary|modify|query|book|emergency|cancel",
    "confidence": float,            # 0-1
    "missing_slots": [str],         # 缺失槽位名列表
}

# ② → ③ 的契约：merged_profile + inferred_slots
profile_output = {
    "user_profile": dict,           # 完整画像
    "preference_vector": list,      # 768维向量
    "inferred_slots": dict,         # 从画像推断的槽位（如预算→历史日均）
    "is_new_user": bool,
}

# ③ → ④ 的契约：Top-15 POI列表
retrieve_output = {
    "poi_candidates": [{            # 最多15个结构化POI
        "spot_id": int,
        "spot_name": str,
        "spot_type": str,           # attraction/restaurant/hotel
        "duration_minutes": int,
        "ticket_price": float,
        "walk_intensity": int,      # 1-5
        "open_time": str,           # "HH:MM"
        "close_time": str,
        "need_reservation": bool,
        "reservation_advance_days": int,
        "queue_time_avg": int,      # 分钟
        "is_peak": bool,
        "indoor_outdoor": str,      # indoor/outdoor/mixed
        "lat": float, "lng": float,
        "tags": [str],
        "_rrf_score": float,        # 融合排序分数
        "_reservation_reminder": bool,  # 需预约的必去景点标记
    }],
    "retrieval_query": str,         # 实际使用的检索query
    "retrieval_empty": bool,        # 是否为空结果
}
```


---

## 1. 编排调度层（第1层）

### 1.1 设计目标

编排调度层是全链路的"交通指挥中心"，负责：
1. **按正确顺序调用6个核心Agent**（understand → profile → retrieve → plan → factcheck → output）
2. **处理用户反馈闭环**（confirm/modify/cancel → 对应重规划或结束）
3. **异常时自动降级**（7类异常 × 每类3级降级）
4. **会话生命周期管理**（创建/续期/超时/恢复）
5. **状态持久化**（PostgreSQL Checkpoint，支持断点恢复）

**关键约束**：编排层只调度，不做业务逻辑。所有LLM调用、DB查询、求解调用都在各Agent内部完成。

---

### 1.2 AgentState 完整定义（全链路状态机）

```python
from typing import TypedDict, Optional, Literal, List, Dict, Any
from datetime import datetime, timedelta

# ===== 辅助类型定义 =====

class Conflict(TypedDict):
    type: str               # budget_insufficient / elderly_too_long / reservation_required / ...
    level: Literal["error", "warning", "info"]
    message: str
    suggestions: List[str]

class FeasibilityReport(TypedDict):
    feasible: bool          # False表示有error级冲突，阻塞流程
    conflicts: List[Conflict]
    conflict_count: int
    error_count: int        # error级冲突数（>0则feasible=False）

class ExecutionTrace(TypedDict):
    node_name: str          # 哪个Node
    started_at: str         # ISO时间
    ended_at: str           # ISO时间
    duration_ms: int        # 耗时
    status: Literal["success", "error", "fallback", "timeout"]
    input_summary: str      # 输入摘要（脱敏）
    output_summary: str     # 输出摘要
    error_message: Optional[str]

class AgentState(TypedDict):
    """
    LangGraph全链路状态机 —— 所有Agent共享的唯一状态对象。
    每次Node执行后返回一个dict，LangGraph自动合并到state中。
    """
    # ═══════════════════════════════════════════
    # 第2层输出：感知层输入
    # ═══════════════════════════════════════════
    user_input: str                     # 用户最新输入（纯文本）
    messages: List[Dict[str, Any]]     # 完整对话历史
    #   格式: [{"role":"system","content":"..."}, {"role":"user","content":"..."},
    #          {"role":"assistant","content":"..."}, ...]
    attachments: List[Dict[str, str]]  # 附件列表 [{"type":"image","path":"/tmp/xx.jpg"}]

    # ═══════════════════════════════════════════
    # 第3层输出：DemandParserAgent
    # ═══════════════════════════════════════════
    slots: Optional[Dict[str, Any]]    # TravelSlots序列化后的dict
    #   完整字段见第3章TravelSlots定义
    missing_slots: List[str]           # 缺失槽位名列表，如["destination","travel_days"]
    intent: Literal["new_itinerary","modify","query","book","emergency","cancel"]
    confidence: float                  # 意图置信度 0.0-1.0，<0.7触发澄清
    feasibility_report: Optional[FeasibilityReport]

    # ═══════════════════════════════════════════
    # 第4层输出：UserProfileRecallAgent
    # ═══════════════════════════════════════════
    user_profile: Optional[Dict[str, Any]]   # user_profile表完整行
    preference_vector: Optional[List[float]] # 768维float列表
    inferred_slots: Dict[str, Any]           # 画像推断的槽位，如{"total_budget": 2000}
    is_new_user: bool                        # 是否首次使用

    # ═══════════════════════════════════════════
    # 第5层输出：TravelRetrievalRAGAgent
    # ═══════════════════════════════════════════
    poi_candidates: List[Dict[str, Any]]     # Top-15结构化POI
    retrieval_query: str                     # 实际发向RAG的query
    retrieval_empty: bool                    # True时触发扩展召回

    # ═══════════════════════════════════════════
    # 第6层输出：ItineraryPlannerAgent
    # ═══════════════════════════════════════════
    itinerary: Optional[List[Dict[str, Any]]]  # 完整行程JSON
    budget_breakdown: Optional[Dict[str, Any]] # 预算明细
    solve_status: Literal["optimal","feasible","infeasible","timeout","fallback"]
    solve_time_ms: int                         # 求解耗时
    conflict_reasons: List[str]                # 不可行时的冲突说明
    replan_mode: Literal["single","local","full","none"]  # v3.0：重规划模式追踪

    # ═══════════════════════════════════════════
    # 第6层输出：FactCheckAgent
    # ═══════════════════════════════════════════
    factcheck_passed: Optional[bool]
    conflicts: List[Dict[str, Any]]            # [{"poi_name":"故宫","field":"open_time","expected":"08:30","actual":"08:00"}]

    # ═══════════════════════════════════════════
    # 第8层输出：OutputFormatAgent
    # ═══════════════════════════════════════════
    output_markdown: Optional[str]             # Markdown文案
    output_pdf_url: Optional[str]              # PDF下载链接
    output_excel_url: Optional[str]            # Excel下载链接

    # ═══════════════════════════════════════════
    # Human-in-the-loop：用户反馈
    # ═══════════════════════════════════════════
    user_feedback: Optional[Dict[str, Any]]
    #   格式: {"action":"replace_poi","day":1,"old_poi_name":"故宫",
    #          "new_poi_id":102,"message":"想换成天坛"}
    feedback_action: Literal["confirm","remove_poi","replace_poi","change_days",
                              "change_budget","change_pace","cancel","none"]

    # ═══════════════════════════════════════════
    # 控制流字段（编排层内部使用）
    # ═══════════════════════════════════════════
    next_node: str                             # 路由决策：下一个Node名
    loop_count: int                            # 当前循环计数（防死循环）
    max_loops: int                             # 最大循环上限（默认5）
    version: int                               # 乐观锁版本号（并发控制）
    created_at: str                            # ISO格式会话创建时间
    updated_at: str                            # ISO格式最后更新时间
    session_timeout_at: str                    # ISO格式会话超时时间（默认30min）
    execution_trace: List[ExecutionTrace]      # 全链路执行轨迹（调试用）

    # ═══════════════════════════════════════════
    # 异常状态
    # ═══════════════════════════════════════════
    error_node: Optional[str]                  # 出错的Node名
    error_message: Optional[str]               # 错误信息
    fallback_used: bool                        # 是否使用了降级方案
    retry_count: int                           # 当前重试次数（FactCheck用）
```

---

### 1.3 LangGraph Node 完整实现

```python
# nodes.py —— 每个Node对应一个Agent的核心调用逻辑

from langchain_core.runnables import RunnableConfig
from typing import Dict, Any
import time

async def understand_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    第3层：DemandParserAgent
    输入：state["user_input"] + state["messages"]
    输出：slots + intent + confidence + feasibility_report + missing_slots
    """
    trace = {"node_name": "understand", "started_at": _now(), "status": "success"}
    try:
        from agents.demand_parser import DemandParserAgent
        parser = DemandParserAgent(config)  # 传入RunnableConfig获取模型配置

        result = await parser.parse(
            user_input=state["user_input"],
            messages=state["messages"],
            attachments=state.get("attachments", [])
        )

        trace.update({"ended_at": _now(), "duration_ms": _elapsed(trace),
                      "input_summary": state["user_input"][:50],
                      "output_summary": f"intent={result['intent']}, slots={len(result['slots'])}}"})

        return {
            "slots": result["slots"],
            "missing_slots": result["missing_slots"],
            "intent": result["intent"],
            "confidence": result["confidence"],
            "feasibility_report": result["feasibility_report"],
            "execution_trace": state.get("execution_trace", []) + [trace]
        }
    except Exception as e:
        trace.update({"ended_at": _now(), "status": "error",
                      "error_message": str(e)[:200]})
        return {
            "error_node": "understand",
            "error_message": str(e),
            "fallback_used": True,
            "execution_trace": state.get("execution_trace", []) + [trace]
        }

async def profile_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    第4层：UserProfileRecallAgent
    输入：state["slots"] + state["messages"]
    输出：user_profile + preference_vector + inferred_slots + is_new_user
    """
    trace = {"node_name": "profile", "started_at": _now(), "status": "success"}
    try:
        from agents.profile_recall import UserProfileRecallAgent
        recall = UserProfileRecallAgent(config)

        user_id = config["configurable"].get("user_id", "anonymous")
        result = await recall.recall(
            user_id=user_id,
            slots=state["slots"],
            messages=state["messages"]
        )

        trace.update({"ended_at": _now(), "duration_ms": _elapsed(trace),
                      "output_summary": f"new_user={result['is_new_user']}"})

        return {
            "user_profile": result["user_profile"],
            "preference_vector": result["preference_vector"],
            "inferred_slots": result["inferred_slots"],
            "is_new_user": result["is_new_user"],
            "execution_trace": state.get("execution_trace", []) + [trace]
        }
    except Exception as e:
        trace.update({"ended_at": _now(), "status": "fallback",
                      "error_message": str(e)[:200]})
        # 降级：使用空画像，不阻塞流程
        return {
            "user_profile": {},
            "preference_vector": None,
            "inferred_slots": {},
            "is_new_user": True,
            "fallback_used": True,
            "execution_trace": state.get("execution_trace", []) + [trace]
        }

async def retrieve_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    第5层：TravelRetrievalRAGAgent
    输入：state["slots"] + state["user_profile"] + state["inferred_slots"]
    输出：poi_candidates + retrieval_query + retrieval_empty
    """
    trace = {"node_name": "retrieve", "started_at": _now(), "status": "success"}
    try:
        from agents.rag_retrieval import TravelRetrievalRAGAgent
        rag = TravelRetrievalRAGAgent(config)

        # 合并slots和画像推断的槽位
        merged_profile = {**(state.get("slots") or {}),
                          **(state.get("inferred_slots") or {})}

        result = await rag.retrieve(
            query=state["user_input"],  # 用户原始输入作为检索query
            profile=merged_profile,
            top_k=15
        )

        trace.update({"ended_at": _now(), "duration_ms": _elapsed(trace),
                      "output_summary": f"retrieved={len(result['poi_candidates'])}"})

        return {
            "poi_candidates": result["poi_candidates"],
            "retrieval_query": result["retrieval_query"],
            "retrieval_empty": result["retrieval_empty"],
            "execution_trace": state.get("execution_trace", []) + [trace]
        }
    except Exception as e:
        trace.update({"ended_at": _now(), "status": "fallback",
                      "error_message": str(e)[:200]})
        return {
            "retrieval_empty": True,
            "poi_candidates": [],
            "fallback_used": True,
            "execution_trace": state.get("execution_trace", []) + [trace]
        }

async def plan_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    第6层：ItineraryPlannerAgent
    输入：state["poi_candidates"] + state["slots"] + state["user_profile"]
    输出：itinerary + solve_status + solve_time_ms + conflict_reasons
    """
    trace = {"node_name": "plan", "started_at": _now(), "status": "success"}
    try:
        from agents.planner import ItineraryPlannerAgent
        planner = ItineraryPlannerAgent(config)

        result = await planner.plan(
            poi_list=state["poi_candidates"],
            slots=state["slots"],
            profile=state.get("user_profile", {})
        )

        trace.update({"ended_at": _now(), "duration_ms": _elapsed(trace),
                      "output_summary": f"status={result['solve_status']}"})

        return {
            "itinerary": result["itinerary"],
            "budget_breakdown": result.get("budget_breakdown"),
            "solve_status": result["solve_status"],
            "solve_time_ms": result["solve_time_ms"],
            "conflict_reasons": result.get("conflict_reasons", []),
            "execution_trace": state.get("execution_trace", []) + [trace]
        }
    except Exception as e:
        trace.update({"ended_at": _now(), "status": "fallback",
                      "error_message": str(e)[:200]})
        return {
            "solve_status": "fallback",
            "fallback_used": True,
            "execution_trace": state.get("execution_trace", []) + [trace]
        }

async def factcheck_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    第6层：FactCheckAgent
    输入：state["itinerary"] + state["poi_candidates"]
    输出：factcheck_passed + conflicts + retry_count
    """
    trace = {"node_name": "factcheck", "started_at": _now(), "status": "success"}
    try:
        from agents.fact_check import FactCheckAgent
        checker = FactCheckAgent(config)

        result = await checker.check(
            itinerary=state["itinerary"],
            poi_reference=state["poi_candidates"]
        )

        trace.update({"ended_at": _now(), "duration_ms": _elapsed(trace),
                      "output_summary": f"passed={result['passed']}"})

        return {
            "factcheck_passed": result["passed"],
            "conflicts": result.get("conflicts", []),
            "retry_count": state.get("retry_count", 0) + (0 if result["passed"] else 1),
            "execution_trace": state.get("execution_trace", []) + [trace]
        }
    except Exception as e:
        trace.update({"ended_at": _now(), "status": "fallback"})
        # 降级：跳过校验，直接通过
        return {
            "factcheck_passed": True,
            "conflicts": [],
            "fallback_used": True,
            "execution_trace": state.get("execution_trace", []) + [trace]
        }

async def output_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    第8层：OutputFormatAgent
    输入：state["itinerary"] + state["user_profile"]
    输出：output_markdown + output_pdf_url + output_excel_url
    """
    trace = {"node_name": "output", "started_at": _now(), "status": "success"}
    try:
        from agents.output_format import OutputFormatAgent
        formatter = OutputFormatAgent(config)

        result = await formatter.format(
            itinerary=state["itinerary"],
            profile=state.get("user_profile", {}),
            budget=state.get("budget_breakdown")
        )

        trace.update({"ended_at": _now(), "duration_ms": _elapsed(trace)})

        return {
            "output_markdown": result["markdown"],
            "output_pdf_url": result.get("pdf_url"),
            "output_excel_url": result.get("excel_url"),
            "execution_trace": state.get("execution_trace", []) + [trace]
        }
    except Exception as e:
        trace.update({"ended_at": _now(), "status": "fallback"})
        # 降级：输出纯文本行程
        return {
            "output_markdown": _generate_fallback_output(state.get("itinerary")),
            "fallback_used": True,
            "execution_trace": state.get("execution_trace", []) + [trace]
        }

# ===== 工具函数 =====

def _now() -> str:
    return datetime.now().isoformat()

def _elapsed(trace: dict) -> int:
    start = datetime.fromisoformat(trace["started_at"])
    return int((datetime.now() - start).total_seconds() * 1000)

def _generate_fallback_output(itinerary):
    """降级输出：纯文本格式化"""
    if not itinerary: return "行程生成失败，请重试。"
    lines = ["# 您的旅行行程\\n"]
    for day in itinerary:
        lines.append(f"## 第{day['day']}天")
        for act in day.get("schedule", []):
            lines.append(f"- {act['arrive_time']} {act['spot_name']}（{act['play_minute']}分钟）")
        lines.append("")
    return "\\n".join(lines)
```

---

### 1.4 条件路由函数（完整分支逻辑）

```python
# routers.py —— 每个条件路由对应一个业务决策点

from typing import Literal

# 路由返回值类型：下游Node的名称
def route_after_understand(state: AgentState) -> Literal[
    "profile",      # 正常：槽位完整且可行 → 继续画像召回
    "output",       # 需要澄清：槽位缺失或约束冲突 → 输出澄清问题
    "retrieve",     # 纯查询：用户只是问信息 → 跳过画像直接检索
    "__end__",      # 取消：用户说"算了""不去了" → 结束
]:
    """
    understand_node执行后的路由决策。
    这是全链路第一个决策点，决定用户是否"表达清楚了一个可规划的需求"。
    """
    intent = state.get("intent", "new_itinerary")
    confidence = state.get("confidence", 0.0)
    feasibility = state.get("feasibility_report")
    missing = state.get("missing_slots", [])

    # 分支1: 用户取消
    if intent == "cancel":
        return "__end__"

    # 分支2: 纯查询（不问行程，只问信息）
    if intent == "query":
        return "retrieve"

    # 分支3: 置信度不足 或 槽位缺失 → 需要澄清
    if confidence < 0.7 or len(missing) > 0:
        return "output"  # 走澄清流程，由output_node生成追问

    # 分支4: 可行性有error级冲突 → 必须让用户确认折中方案
    if feasibility and not feasibility["feasible"]:
        return "output"  # 输出冲突说明和折中建议

    # 分支5: 正常流程
    return "profile"


def route_after_planner(state: AgentState) -> Literal[
    "factcheck",      # 正常：有可行解 → 进入事实校验
    "output",         # 不可行：求解失败 → 输出冲突说明
    "human_review",   # 重试超限：FactCheck失败3次 → 人工审核
]:
    """
    plan_node执行后的路由决策。
    判断求解结果的质量，决定是否进入校验环节。
    """
    solve_status = state.get("solve_status", "infeasible")
    retry = state.get("retry_count", 0)

    if solve_status == "infeasible":
        return "output"  # 输出不可行原因和折中方案

    if retry >= 3:
        return "human_review"  # 转人工

    return "factcheck"


def route_after_factcheck(state: AgentState) -> Literal[
    "output",    # 校验通过 或 已重试3次 → 输出给用户
    "plan",      # 校验不通过且未超限 → 回到Planner重规划
]:
    """
    factcheck_node执行后的路由决策。
    循环控制：最多3次重规划。
    """
    passed = state.get("factcheck_passed", False)
    retry = state.get("retry_count", 0)

    if passed or retry >= 3:
        return "output"

    return "plan"  # 回到plan_node重新求解


def route_after_output(state: AgentState) -> Literal[
    "apply_single_change",  # 单点替换/删除 → 直接修改行程
    "replan_local",         # 局部调整 → 单日重算
    "plan",                 # 全局变更 → 全量重算
    "__end__",              # 确认满意 或 取消 → 结束
    "human_interrupt",      # 等待用户反馈 → Human-in-the-loop
]:
    """
    output_node执行后的路由决策（Human-in-the-loop）。
    这是最关键的分支点：根据用户的反馈动作决定下一步。
    """
    action = state.get("feedback_action", "none")

    if action == "confirm":
        # 用户满意 → 可选Booking → 更新画像 → 结束
        return "__end__"

    if action == "cancel":
        return "__end__"

    if action in ("remove_poi", "replace_poi"):
        return "apply_single_change"  # O(n)直接修改

    if action in ("change_days", "change_budget"):
        return "plan"  # 全局变更，全量重算

    if action in ("change_pace",):
        return "replan_local"  # 局部变更，单日重算

    # 默认：等待用户反馈
    return "human_interrupt"
```

---

### 1.5 Graph 构建（完整版）

```python
# graph.py —— LangGraph StateGraph 完整构建

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver

# 导入所有Node
from nodes import (
    understand_node, profile_node, retrieve_node,
    plan_node, factcheck_node, output_node,
    apply_single_change_node, replan_local_node,
    error_handler_node, human_interrupt_node
)
from routers import (
    route_after_understand, route_after_planner,
    route_after_factcheck, route_after_output
)

def build_travel_graph():
    """构建完整的旅游Agent LangGraph"""
    graph = StateGraph(AgentState)

    # 注册所有Node
    graph.add_node("understand",      understand_node)
    graph.add_node("profile",         profile_node)
    graph.add_node("retrieve",        retrieve_node)
    graph.add_node("plan",            plan_node)
    graph.add_node("factcheck",       factcheck_node)
    graph.add_node("output",          output_node)
    graph.add_node("apply_single",    apply_single_change_node)
    graph.add_node("replan_local",    replan_local_node)
    graph.add_node("error_handler",   error_handler_node)
    graph.add_node("human_interrupt", human_interrupt_node)

    # 设置入口点
    graph.set_entry_point("understand")

    # 条件边：understand → profile / output / retrieve / end
    graph.add_conditional_edges(
        "understand",
        route_after_understand,
        {
            "profile": "profile",
            "output": "output",
            "retrieve": "retrieve",
            "__end__": END,
        }
    )

    # 顺序边：profile → retrieve → plan
    graph.add_edge("profile", "retrieve")
    graph.add_edge("retrieve", "plan")

    # 条件边：plan → factcheck / output / human_review
    graph.add_conditional_edges(
        "plan",
        route_after_planner,
        {
            "factcheck": "factcheck",
            "output": "output",
            "human_review": "human_interrupt",
        }
    )

    # 条件边：factcheck → output / plan(重试)
    graph.add_conditional_edges(
        "factcheck",
        route_after_factcheck,
        {
            "output": "output",
            "plan": "plan",
        }
    )

    # 条件边：output → apply_single / replan_local / plan / end / human_interrupt
    graph.add_conditional_edges(
        "output",
        route_after_output,
        {
            "apply_single_change": "apply_single",
            "replan_local": "replan_local",
            "plan": "plan",
            "__end__": END,
            "human_interrupt": "human_interrupt",
        }
    )

    # 修改类Node完成后回到output
    graph.add_edge("apply_single", "output")
    graph.add_edge("replan_local", "output")

    # HumanInterrupt等待用户输入后回到understand
    graph.add_edge("human_interrupt", "understand")

    # Checkpoint持久化
    checkpointer = PostgresSaver(conn_string="postgresql://user:pass@localhost/travel_agent")

    return graph.compile(checkpointer=checkpointer)

# 全局实例
travel_graph = build_travel_graph()
```

---

### 1.6 异常处理：7类异常 × 3级降级

```python
# exceptions.py —— 全链路异常处理与降级策略

class NodeException(Exception):
    """Node执行异常基类"""
    def __init__(self, node_name: str, original_error: Exception, severity: str):
        self.node_name = node_name
        self.original = original_error
        self.severity = severity  # critical / warning / info
        super().__init__(f"[{node_name}] {str(original_error)}")

async def global_error_handler(state: AgentState, exception: NodeException) -> Dict[str, Any]:
    """
    全局异常处理器 —— 7类异常 × 3级降级
    原则：用户无感知降级，关键错误才阻断流程
    """
    node = exception.node_name
    severity = exception.severity

    # ===== 7类异常的标准化降级策略 =====
    strategies = {
        # ── 1. 感知层异常 ──
        "understand": {
            "critical": {  # LLM完全不返回
                "output": "抱歉，我没能理解您的意思。请用简洁的语言描述您的旅行需求，\n"
                         "比如：\"我想带父母去北京玩3天，预算5000元。\"",
                "fallback_used": True,
            },
            "warning": {  # 槽位抽取不全
                "output": "我理解了您的部分需求。为了更好地为您规划，请告诉我：\n"
                         + _format_missing_slots(state.get("missing_slots", [])),
                "fallback_used": True,
            },
        },

        # ── 2. 画像层异常 ──
        "profile": {
            "critical": {  # DB完全不可用
                "user_profile": {},           # 空画像
                "preference_vector": None,
                "is_new_user": True,
                "fallback_used": True,
            },
            "warning": {  # 向量检索失败
                "user_profile": state.get("user_profile", {}),  # 保留已有
                "preference_vector": None,
                "fallback_used": True,
            },
        },

        # ── 3. 检索层异常 ──
        "retrieve": {
            "critical": {  # pgvector + Redis都不可用
                "poi_candidates": [],         # 空列表
                "retrieval_empty": True,
                "output": "抱歉，目的地信息暂时无法获取。请稍后再试，\n"
                         "或告诉我具体想去的城市，我为您手动查询。",
                "fallback_used": True,
            },
            "warning": {  # 检索结果为空
                "poi_candidates": [],
                "retrieval_empty": True,
                "output": f"抱歉，我在「{state.get('slots', {}).get('destination', '该城市')}」"
                         f"没有找到符合条件的景点。请尝试：\n"
                         f"1. 换一个目的地\n2. 放宽预算或天数\n3. 减少饮食禁忌",
                "fallback_used": True,
            },
        },

        # ── 4. 规划层异常 ──
        "plan": {
            "critical": {  # CP-SAT + 贪心都失败
                "solve_status": "fallback",
                "itinerary": None,
                "output": "行程规划暂时不可用，工程师正在修复。\n"
                         "您可以先告诉我必去的景点，我帮您手动排顺序。",
                "fallback_used": True,
            },
            "warning": {  # CP-SAT超时但贪心成功
                "solve_status": "fallback",
                "fallback_used": True,
                # 行程已由贪心生成，直接输出
            },
        },

        # ── 5. 校验层异常 ──
        "factcheck": {
            "critical": {  # DB校验完全失败
                "factcheck_passed": True,     # 跳过校验，直接通过
                "conflicts": [],
                "output": "行程已生成（数据校验服务暂时不可用，建议出行前再次确认景点开放时间）。",
                "fallback_used": True,
            },
        },

        # ── 6. 输出层异常 ──
        "output": {
            "critical": {  # LLM格式化失败
                "output_markdown": _generate_plaintext_itinerary(state.get("itinerary")),
                "fallback_used": True,
            },
        },

        # ── 7. 编排层异常 ──
        "orchestrator": {
            "critical": {  # 状态机损坏
                "output": "系统出现内部错误，请重新开始对话。",
                "fallback_used": True,
            },
        },
    }

    node_strategies = strategies.get(node, {})
    level_key = "critical" if severity == "critical" else "warning"
    fallback = node_strategies.get(level_key, {"fallback_used": True})

    # 记录到trace
    trace = state.get("execution_trace", [])
    trace.append({
        "node_name": f"{node}_error_handler",
        "started_at": datetime.now().isoformat(),
        "status": "fallback",
        "error_message": str(exception.original)[:200],
    })
    fallback["execution_trace"] = trace

    return fallback


def _format_missing_slots(slots: list) -> str:
    """格式化缺失槽位为追问文案"""
    slot_names = {
        "destination": "目的地（想去哪个城市）",
        "travel_days": "旅行天数",
        "travel_dates": "出发日期",
        "total_budget": "总预算",
        "travelers_count": "出行人数",
    }
    return "\n".join([f"- {slot_names.get(s, s)}" for s in slots])


def _generate_plaintext_itinerary(itinerary):
    """纯文本降级输出"""
    if not itinerary: return "行程生成失败，请重试。"
    lines = ["# 您的旅行行程\\n"]
    for day in itinerary:
        lines.append(f"## 第{day['day']}天")
        for act in day.get("schedule", []):
            lines.append(f"- {act.get('arrive_time', '?')} {act['spot_name']} "
                        f"（游玩{act.get('play_minute', '?')}分钟）")
        lines.append("")
    return "\\n".join(lines)
```

---

### 1.7 Human-in-the-loop 交互设计

```python
# human_interrupt.py —— 用户反馈处理Node

async def human_interrupt_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Human-in-the-loop节点：展示行程后等待用户反馈。
    通过SSE流式推送output_markdown到前端，前端展示后提供操作按钮。
    """
    trace = {"node_name": "human_interrupt", "started_at": _now(), "status": "success"}

    # 生成可操作的反馈选项
    feedback_options = _generate_feedback_options(state)

    trace.update({"ended_at": _now(), "duration_ms": _elapsed(trace)})

    return {
        "output_markdown": state.get("output_markdown", "") + "\n\n" + feedback_options,
        "execution_trace": state.get("execution_trace", []) + [trace]
    }


def _generate_feedback_options(state: AgentState) -> str:
    """根据当前状态生成用户可操作选项"""
    options = [
        "\\n---",
        "\\n**您对这份行程满意吗？**\\n",
        "1. ✅ 确认满意，生成PDF",
        "2. 🔄 调整某天的顺序",
        "3. ➕ 增加一个景点",
        "4. ➖ 删除某个景点",
        "5. 🏨 替换某个景点",
        "6. 📅 调整天数或预算",
        "7. ❌ 取消",
        "\\n*直接输入您的修改要求也可以，比如：\"第2天不去故宫了，换成天坛\"*",
    ]
    return "\\n".join(options)


async def apply_single_change_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    单点修改Node：直接修改行程JSON，不调用求解器。
    O(n)复杂度，即时响应。
    """
    feedback = state.get("user_feedback", {})
    action = feedback.get("action", "")
    itinerary = state.get("itinerary", [])

    if action == "remove_poi":
        day_idx = feedback.get("day", 1) - 1
        poi_name = feedback.get("poi_name")
        if 0 <= day_idx < len(itinerary):
            itinerary[day_idx]["schedule"] = [
                a for a in itinerary[day_idx]["schedule"]
                if a["spot_name"] != poi_name
            ]

    elif action == "replace_poi":
        day_idx = feedback.get("day", 1) - 1
        old_name = feedback.get("old_poi_name")
        new_id = feedback.get("new_poi_id")
        poi_list = state.get("poi_candidates", [])

        try:
            new_poi = next(p for p in poi_list if p["spot_id"] == new_id)
        except StopIteration:
            return {"itinerary": itinerary, "error_message": f"找不到景点ID {new_id}"}

        if 0 <= day_idx < len(itinerary):
            for act in itinerary[day_idx]["schedule"]:
                if act["spot_name"] == old_name:
                    act["spot_id"] = new_poi["spot_id"]
                    act["spot_name"] = new_poi["spot_name"]
                    act["play_minute"] = new_poi.get("duration_minutes", 120)
                    act["ticket_cost"] = new_poi.get("ticket_price", 0)
                    act["walk_score"] = new_poi.get("walk_intensity", 1)
                    break

    return {"itinerary": itinerary, "replan_mode": "single"}


async def replan_local_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    局部重规划Node：只重算目标日期，固定其他天。
    调用CP-SAT时D=1，大幅降低求解复杂度。
    """
    feedback = state.get("user_feedback", {})
    target_day = feedback.get("day", 1) - 1
    constraints = state.get("slots", {})
    poi_list = state.get("poi_candidates", [])
    current_itinerary = state.get("itinerary", [])

    # 1. 收集已固定的POI（其他天的景点不能动）
    fixed_pois = set()
    for d, day in enumerate(current_itinerary):
        if d == target_day: continue
        for act in day.get("schedule", []):
            fixed_pois.add(act["spot_id"])

    # 2. 可用POI = 候选池 - 已固定
    available = [p for p in poi_list if p["spot_id"] not in fixed_pois]

    # 3. 构建单日约束
    day_constraints = {
        **constraints,
        "D": 1,  # 只求解1天
    }
    # 老人模式：降低步行上限
    if constraints.get("has_elderly"):
        day_constraints["Walk_max"] = constraints.get("Walk_max", 10) - 2

    # 4. 调用求解器（D=1，小规模直接用贪心）
    from agents.planner import ItineraryPlannerAgent
    planner = ItineraryPlannerAgent(config)
    result = await planner.plan(available, day_constraints, state.get("user_profile", {}))

    # 5. 合并回原行程
    new_itin = list(current_itinerary)
    if result.get("itinerary") and len(result["itinerary"]) > 0:
        new_itin[target_day] = result["itinerary"][0]

    return {"itinerary": new_itin, "replan_mode": "local"}
```

---

### 1.8 会话生命周期管理

```python
# session_manager.py —— 会话创建、续期、超时、恢复

from datetime import datetime, timedelta
import uuid

class SessionManager:
    """会话生命周期管理"""

    SESSION_TTL_MINUTES = 30  # 30分钟无操作自动过期

    async def create_session(self, user_id: str) -> str:
        """创建新会话，返回session_id"""
        session_id = f"sess_{uuid.uuid4().hex[:16]}"
        now = datetime.now()
        initial_state: AgentState = {
            "user_input": "",
            "messages": [],
            "attachments": [],
            "slots": None,
            "missing_slots": [],
            "intent": "new_itinerary",
            "confidence": 0.0,
            "feasibility_report": None,
            "user_profile": None,
            "preference_vector": None,
            "inferred_slots": {},
            "is_new_user": True,
            "poi_candidates": [],
            "retrieval_query": "",
            "retrieval_empty": False,
            "itinerary": None,
            "budget_breakdown": None,
            "solve_status": "none",
            "solve_time_ms": 0,
            "conflict_reasons": [],
            "replan_mode": "none",
            "factcheck_passed": None,
            "conflicts": [],
            "output_markdown": None,
            "output_pdf_url": None,
            "output_excel_url": None,
            "user_feedback": None,
            "feedback_action": "none",
            "next_node": "understand",
            "loop_count": 0,
            "max_loops": 5,
            "version": 1,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "session_timeout_at": (now + timedelta(minutes=self.SESSION_TTL_MINUTES)).isoformat(),
            "execution_trace": [],
            "error_node": None,
            "error_message": None,
            "fallback_used": False,
            "retry_count": 0,
        }
        # 写入PostgreSQL Checkpoint
        await self._save_checkpoint(session_id, initial_state)
        return session_id

    async def resume_session(self, session_id: str) -> Optional[AgentState]:
        """从Checkpoint恢复会话。如果已超时，保留草稿但标记为过期。"""
        state = await self._load_checkpoint(session_id)
        if not state:
            return None

        timeout = datetime.fromisoformat(state["session_timeout_at"])
        if datetime.now() > timeout:
            state["fallback_used"] = True
            state["error_message"] = "会话已超时（30分钟无操作），已为您保留草稿。"
            # 不删除，让用户可以继续

        # 续期
        state["session_timeout_at"] = (
            datetime.now() + timedelta(minutes=self.SESSION_TTL_MINUTES)
        ).isoformat()
        state["version"] += 1

        return state

    async def _save_checkpoint(self, session_id: str, state: AgentState):
        """持久化到PostgreSQL"""
        # 通过LangGraph PostgresSaver自动完成
        pass

    async def _load_checkpoint(self, session_id: str) -> Optional[AgentState]:
        """从PostgreSQL恢复"""
        # 通过LangGraph PostgresSaver自动完成
        pass
```


---

## 2. 感知层（第2层）

### 2.1 职责定位

感知层是全链路的"感官系统"，负责将用户的各种输入（文字、图片、PDF、外部事件）统一转化为**纯文本 + 附件元数据**，供第3层（意图理解层）处理。

**核心原则**：感知层只做"格式转换"，不做"语义理解"。语义理解留给DemandParserAgent（第3层）。

---

### 2.2 四类输入处理

#### 2.2.1 文本输入（SSE流式对话）

```python
# text_input.py —— FastAPI SSE文本接收

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import json
import asyncio

app = FastAPI()

@app.post("/api/v1/chat")
async def chat_endpoint(request: Request):
    """SSE流式对话入口"""
    body = await request.json()
    session_id = body.get("session_id")
    user_input = body.get("message", "").strip()
    attachments = body.get("attachments", [])  # [{"type": "image", "url": "..."}]

    if not user_input and not attachments:
        return {"error": "请输入内容或上传附件"}

    async def event_stream():
        """SSE流：逐字输出LLM生成的回复"""
        # 1. 构建初始状态
        state = await session_manager.resume_session(session_id)
        if not state:
            session_id_new = await session_manager.create_session(
                body.get("user_id", "anonymous")
            )
            state = await session_manager.resume_session(session_id_new)

        # 2. 更新用户输入
        state["user_input"] = user_input
        state["messages"].append({"role": "user", "content": user_input})
        state["attachments"] = attachments

        # 3. 调用LangGraph（流式输出）
        async for event in travel_graph.astream_events(state, version="v1"):
            if event["event"] == "on_chat_model_stream":
                # 流式输出LLM生成的token
                token = event["data"]["chunk"].content
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            elif event["event"] == "on_chain_end":
                # 最终状态输出
                final_state = event["data"]["output"]
                yield f"data: {json.dumps({'type': 'final', 'markdown': final_state.get('output_markdown')})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

**SSE事件类型定义**：

| 事件类型 | 触发时机 | 前端行为 |
|---------|---------|---------|
| `token` | LLM生成每个token时 | 逐字渲染到对话框 |
| `thinking` | Agent开始思考时 | 显示"思考中..."动画 |
| `tool_call` | 调用外部工具时 | 显示工具名称和参数 |
| `partial` | 中间状态更新时 | 显示部分结果（如POI列表） |
| `final` | 全链路完成时 | 渲染最终Markdown + 操作按钮 |
| `error` | 异常发生时 | 显示错误提示 + 重试按钮 |
| `clarify` | 需要用户补充时 | 显示追问表单 |

---

#### 2.2.2 图片输入（MVP降级方案：PaddleOCR）

```python
# image_input.py —— 图片处理（MVP不用Qwen-VL）

class ImageProcessor:
    """
    MVP阶段：使用PaddleOCR提取文字，不调用Qwen2.5-VL。
    原因：VL模型需要GPU，MVP阶段降低硬件依赖。

    升级路径：
      MVP: PaddleOCR → 提取文字 → DemandParser解析文字
      P4:  Qwen2.5-VL-7B → 图片理解 → 直接结构化输出
    """

    def __init__(self):
        self.ocr = None  # 懒加载

    def _get_ocr(self):
        if self.ocr is None:
            from paddleocr import PaddleOCR
            # use_gpu=False 确保CPU环境可运行
            self.ocr = PaddleOCR(
                use_angle_cls=True,    # 方向分类（处理旋转图片）
                lang="ch",             # 中文
                use_gpu=False,         # MVP：CPU运行
                show_log=False,        # 关闭冗余日志
            )
        return self.ocr

    async def process(self, image_path: str) -> dict:
        """
        处理图片：OCR提取文字 + 基础元数据
        返回：{"text": "提取的文字", "confidence": 0.95, "word_count": 42}
        """
        try:
            ocr = self._get_ocr()
            result = ocr.ocr(image_path, cls=True)

            if not result or not result[0]:
                return {"text": "", "confidence": 0.0, "word_count": 0}

            # 提取文字和置信度
            lines = []
            total_conf = 0.0
            for line in result[0]:
                bbox, (text, conf) = line
                lines.append(text)
                total_conf += conf

            full_text = "\\n".join(lines)
            avg_conf = total_conf / len(lines) if lines else 0.0

            return {
                "text": full_text[:2000],  # 限制长度，避免超出LLM上下文
                "confidence": round(avg_conf, 2),
                "word_count": len(full_text),
                "source": "paddleocr",
            }

        except Exception as e:
            # 降级：OCR失败时返回提示
            return {
                "text": "[图片处理失败，请直接输入文字描述您的需求]",
                "confidence": 0.0,
                "word_count": 0,
                "source": "fallback",
                "error": str(e)[:100]
            }

    async def process_upload(self, file_bytes: bytes, filename: str) -> dict:
        """处理前端上传的图片文件"""
        import tempfile, os
        suffix = os.path.splitext(filename)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            return await self.process(tmp_path)
        finally:
            os.unlink(tmp_path)
```

**PaddleOCR vs Qwen-VL 对比**：

| 维度 | PaddleOCR (MVP) | Qwen2.5-VL (P4+) |
|------|----------------|------------------|
| 功能 | 仅文字提取 | 图片理解+结构化输出 |
| GPU需求 | 否（CPU可跑） | 是（需A10/4090） |
| 准确率 | 95%+（印刷体） | 理解能力更强 |
| 延迟 | <2s（CPU） | 3-5s（GPU） |
| 成本 | ¥0 | ~¥500/月（GPU） |
| 场景 | 攻略截图、门票照片 | 景点照片识别、地图理解 |

---

#### 2.2.3 PDF输入（PyMuPDF文字提取）

```python
# pdf_input.py —— PDF攻略文档处理

import fitz  # PyMuPDF

class PDFProcessor:
    """PDF攻略文档处理器：提取文字+表格+图片说明"""

    async def process(self, pdf_path: str) -> dict:
        """
        处理PDF：提取文字内容，限制长度。
        不尝试理解内容，只提取纯文本交给DemandParser。
        """
        try:
            doc = fitz.open(pdf_path)
            text_parts = []

            for page_num, page in enumerate(doc):
                text = page.get_text()
                # 清理多余空白
                text = " ".join(text.split())
                text_parts.append(f"[第{page_num+1}页]\\n{text}")

                # 最多处理前20页（避免超大PDF）
                if page_num >= 19:
                    text_parts.append("[文档过长，仅提取前20页]")
                    break

            full_text = "\\n\\n".join(text_parts)

            return {
                "text": full_text[:5000],  # 限制5KB
                "page_count": len(doc),
                "processed_pages": min(len(doc), 20),
                "source": "pymupdf",
            }

        except Exception as e:
            return {
                "text": "[PDF处理失败，请直接输入文字描述]",
                "source": "fallback",
                "error": str(e)[:100]
            }

    async def extract_tables(self, pdf_path: str) -> list:
        """提取PDF中的表格（攻略中的行程表、预算表）"""
        # 使用camelot-py或tabula-py提取表格
        # MVP阶段暂不实现，P2后补充
        return []
```

---

#### 2.2.4 外部事件输入（Webhook）

```python
# webhook_handler.py —— 外部事件处理

from fastapi import FastAPI, BackgroundTasks
from datetime import datetime

app = FastAPI()

class WebhookHandler:
    """
    外部事件处理器：接收天气/交通/景区变更事件，
    触发局部重规划或用户通知。
    """

    # 事件类型注册表
    EVENT_HANDLERS = {
        "weather_alert": "handle_weather_alert",      # 天气预警
        "attraction_closed": "handle_attraction_closed",  # 景区闭园
        "traffic_delay": "handle_traffic_delay",      # 交通延误
        "flight_changed": "handle_flight_changed",    # 航班变更
        "user_manual_update": "handle_user_update",   # 用户主动修改
    }

    async def handle(self, event_type: str, payload: dict) -> dict:
        """分发事件到对应处理器"""
        handler_name = self.EVENT_HANDLERS.get(event_type)
        if not handler_name:
            return {"status": "ignored", "reason": f"未知事件类型: {event_type}"}

        handler = getattr(self, handler_name)
        return await handler(payload)

    async def handle_weather_alert(self, payload: dict) -> dict:
        """
        天气预警处理：
        1. 获取受影响的用户会话列表
        2. 判断预警是否影响行程（暴雨→室外景点）
        3. 推送通知给用户
        4. 提供一键替换方案
        """
        city = payload.get("city")
        alert_level = payload.get("level")  # blue/yellow/orange/red
        weather = payload.get("weather")    # rain/storm/snow/heat

        # 只有yellow及以上才处理
        if alert_level not in ("yellow", "orange", "red"):
            return {"status": "ignored", "reason": "预警级别不足"}

        # 获取该城市下正在进行的行程
        affected_sessions = await self._get_active_sessions(city)

        for session in affected_sessions:
            # 判断行程中是否有室外景点
            has_outdoor = any(
                act.get("indoor_outdoor") == "outdoor"
                for day in session.get("itinerary", [])
                for act in day.get("schedule", [])
            )

            if has_outdoor:
                # 推送通知
                await self._push_notification(
                    session["user_id"],
                    f"【天气预警】{city}发布{alert_level}预警，"
                    f"您的行程中有室外景点受影响。建议一键替换为室内景点。"
                )
                # 标记待重规划
                await self._mark_for_replan(session["session_id"], "weather", payload)

        return {"status": "processed", "affected_sessions": len(affected_sessions)}

    async def handle_attraction_closed(self, payload: dict) -> dict:
        """景区临时闭园处理"""
        spot_id = payload.get("spot_id")
        spot_name = payload.get("spot_name")
        closure_dates = payload.get("closure_dates", [])  # ["2026-07-01", "2026-07-02"]

        # 获取包含该景点的行程
        affected = await self._get_sessions_with_spot(spot_id)

        for session in affected:
            await self._push_notification(
                session["user_id"],
                f"【景区通知】「{spot_name}」于{', '.join(closure_dates)}临时闭园，"
                f"已为您推荐替代景点，请查看。"
            )
            await self._mark_for_replan(session["session_id"], "closure", payload)

        return {"status": "processed", "affected_sessions": len(affected)}

    async def _mark_for_replan(self, session_id: str, reason: str, payload: dict):
        """标记会话需要重规划"""
        # 写入Redis队列，由调度器定期消费
        import redis
        r = redis.Redis()
        r.lpush("replan_queue", json.dumps({
            "session_id": session_id,
            "reason": reason,
            "payload": payload,
            "created_at": datetime.now().isoformat()
        }))


@app.post("/api/v1/webhooks/events")
async def receive_webhook(request, background_tasks: BackgroundTasks):
    """外部事件Webhook入口"""
    body = await request.json()
    event_type = body.get("event_type")
    payload = body.get("payload", {})

    handler = WebhookHandler()
    background_tasks.add_task(handler.handle, event_type, payload)

    return {"status": "accepted", "event_type": event_type}
```

---

### 2.3 感知层输出数据契约

感知层处理完成后，输出统一格式的数据供第3层消费：

```python
# 感知层输出 = 第3层输入
PerceptionOutput = {
    "user_input": str,          # 用户输入的纯文本（OCR/PDF提取的文字拼接在这里）
    "messages": list,           # 完整对话历史
    "attachments_meta": [{      # 附件元数据（不传递文件内容）
        "type": "image|pdf",
        "source": "paddleocr|pymupdf|qwen-vl",
        "extracted_text": str,  # 提取的文字内容
        "confidence": float,    # 提取置信度
        "original_url": str,   # 文件存储路径
    }],
    "external_event": {         # 外部事件（如有）
        "type": "weather_alert|attraction_closed|...",
        "payload": dict,
    } or None,
}
```


---

## 3. 意图理解层（第3层）

### 3.1 职责定位

意图理解层是"需求分析师"，核心任务：
1. **槽位抽取**：从用户自然语言中提取15个结构化字段（目的地、天数、预算等）
2. **意图分类**：判断用户是新建行程/修改行程/查询信息/取消
3. **情感识别**：检测用户情绪（焦虑/兴奋/犹豫），影响后续文案语气
4. **歧义消解**：当信息不完整时，通过追问澄清
5. **可行性前置校验**：在调用求解器前，检查需求是否合理

**关键设计**：DemandParserAgent = LLM槽位抽取 + 规则引擎校验，两层把关。

---

### 3.2 DemandParserAgent 完整实现

```python
# agents/demand_parser.py —— 需求解析Agent

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import Optional, Literal, List
import json

class TravelSlots(BaseModel):
    """旅行需求槽位 —— 15个字段，覆盖完整旅行规划参数"""

    # ── 基础信息（4个）──
    origin: Optional[str] = Field(None, description="出发地城市，如'上海'")
    destination: Optional[str] = Field(None, description="目的地城市，如'北京'")
    travel_days: Optional[int] = Field(None, description="旅行天数，如3")
    travel_dates: Optional[str] = Field(None, description="日期范围，如'2026-07-01~2026-07-03'")

    # ── 人群信息（6个）──
    travelers_count: int = Field(1, description="出行人数")
    has_elderly: bool = Field(False, description="是否有65岁以上老人")
    has_children: bool = Field(False, description="是否有12岁以下儿童")
    has_pregnant: bool = Field(False, description="是否有孕妇")
    has_wheelchair: bool = Field(False, description="是否有轮椅使用者")
    travel_companion: Literal["solo","couple","family_kid","family_elder",
                               "friends","business"] = Field("solo", description="出行同伴类型")

    # ── 预算（2个）──
    total_budget: Optional[float] = Field(None, description="总预算人民币，如5000")
    budget_per_person: Optional[float] = Field(None, description="人均预算")

    # ── 偏好（5个）──
    interests: List[str] = Field([], description="兴趣标签，如['历史','美食','拍照']")
    food_prefs: List[str] = Field([], description="饮食偏好，如['辣','清淡']")
    food_taboos: List[str] = Field([], description="饮食禁忌，如['不吃辣','海鲜过敏','清真']")
    must_visit: List[str] = Field([], description="必去景点白名单，如['故宫','长城']")
    must_not_visit: List[str] = Field([], description="不去景点黑名单")

    # ── 约束（9个）──
    pace: Literal["relaxed","moderate","intensive"] = Field("moderate", description="节奏: relaxed(轻松)/moderate(适中)/intensive(紧凑)")
    play_mode: Literal["quick","standard","deep"] = Field("standard", description="游玩深度: quick(快速打卡)/standard(标准)/deep(深度体验)。影响CP-SAT场景参数计算")
    max_walk_minutes: int = Field(180, description="日最大步行分钟数")
    max_transit_minutes: int = Field(120, description="日最大车程分钟数")
    avoid_crowds: bool = Field(False, description="是否避开人流")
    prefer_morning: bool = Field(False, description="是否早起型")
    include_restaurant: bool = Field(False, description="是否安排餐厅（opt-in，默认不安排）")
    transport_preference: List[str] = Field(["subway","walk","taxi"], description="交通偏好排序")
    fatigue_preference: Optional[str] = Field(None, description="疲劳偏好: low(轻松)/medium(适中)/high(紧凑)。影响CP-SAT疲劳恢复系数")

    # 注意：CP-SAT求解参数（T_day_max/Walk_max/Rest_day/Budget_day_max/Drive_max/food_day等）
    # 由CPSATTuningGuide根据人群类型+节奏+目的地自动计算，不在需求层暴露。
    # 详见下半部分第6章CPSATTuningGuide.tune()方法。


class DemandParserAgent:
    """
    需求解析Agent：从用户自然语言中提取结构化旅行需求。

    处理流程：
      1. 构建Prompt（系统指令 + 对话历史 + 当前输入）
      2. 调用LLM（7B-AWQ，轻量快速）
      3. 解析JSON输出为TravelSlots
      4. 规则引擎校验（必填项检查 + 冲突检测）
      5. 可行性校验（预算vs城市成本 + 人群vs行程长度）
      6. 返回结果 + 缺失项 + 追问文案
    """

    # LLM Prompt模板
    SYSTEM_PROMPT = """你是一个专业的旅行需求分析助手。你的任务是从用户的自然语言描述中，
提取结构化的旅行规划参数。请严格按JSON格式输出，不要添加任何解释文字。

可识别的意图类型：
- new_itinerary: 用户想要一份新行程
- modify: 用户想修改已有行程
- query: 用户只是询问信息（不需要规划）
- cancel: 用户取消当前操作

可识别的情感状态：
- excited: 用户很兴奋（如"好期待！"）
- anxious: 用户焦虑担忧（如"会不会很累？"）
- hesitant: 用户犹豫不决（如"不知道去哪好"）
- neutral: 中性

输出JSON格式：
{
  "intent": "new_itinerary|modify|query|cancel",
  "confidence": 0.0-1.0,
  "sentiment": "excited|anxious|hesitant|neutral",
  "slots": {
    "destination": "目的地城市或null",
    "travel_days": 天数或null,
    "travel_dates": "日期范围或null",
    "travelers_count": 人数,
    "has_elderly": true/false,
    "has_children": true/false,
    "has_pregnant": true/false,
    "has_wheelchair": true/false,
    "travel_companion": "solo|couple|family_kid|family_elder|friends|business",
    "total_budget": 总预算数字或null,
    "interests": ["兴趣标签"],
    "food_prefs": ["饮食偏好"],
    "food_taboos": ["饮食禁忌"],
    "must_visit": ["必去景点"],
    "must_not_visit": ["不去景点"],
    "pace": "relaxed|moderate|intensive",
    "transport_preference": ["subway","walk","taxi"]
  },
  "missing_slots": ["缺失的必填项名称"],
  "clarifying_question": "如果需要追问，生成自然语言问题"
}

规则：
1. 只输出JSON，不输出任何其他文字
2. 不确定的字段填null，不要猜测
3. must_visit中的景点名尽量使用官方名称
4. 如果用户说"带爸妈"，则has_elderly=true, travel_companion=family_elder
5. 如果用户说"带孩子"，则has_children=true, travel_companion=family_kid
6. 预算如果没有明确说"人均"，默认是总预算
7. 情感检测：关注用户的语气词（"好期待""担心""不知道"）
"""

    def __init__(self, config: RunnableConfig):
        self.config = config
        # 使用7B-AWQ模型（轻量快速）
        self.llm = ChatOpenAI(
            model="Qwen2.5-7B-AWQ",
            base_url="http://vllm-7b:8000/v1",
            api_key="dummy",
            temperature=0.1,  # 低温度，稳定输出
            max_tokens=1024,
        )
        self.parser = JsonOutputParser()
        self.feasibility_checker = FeasibilityChecker()

    async def parse(self, user_input: str, messages: list, attachments: list) -> dict:
        # 步骤1：构建Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            *[(m["role"], m["content"]) for m in messages[-10:]],  # 最近10轮
            ("human", user_input),
        ])

        # 步骤2：调用LLM
        chain = prompt | self.llm | self.parser
        raw_result = await chain.ainvoke({}, config=self.config)

        # 步骤3：解析为TravelSlots
        slots_data = raw_result.get("slots", {})
        slots = TravelSlots(**slots_data)

        # 步骤4：规则引擎校验 —— 必填项检查
        missing = self._check_required(slots, raw_result.get("missing_slots", []))

        # 步骤5：可行性校验
        feasibility = self.feasibility_checker.check(slots)

        # 步骤6：生成追问文案（如果有缺失项）
        clarify_question = self._generate_clarify_question(slots, missing, feasibility)

        return {
            "slots": slots.model_dump(),
            "intent": raw_result.get("intent", "new_itinerary"),
            "confidence": raw_result.get("confidence", 0.5),
            "sentiment": raw_result.get("sentiment", "neutral"),
            "missing_slots": missing,
            "feasibility_report": feasibility,
            "clarifying_question": clarify_question,
        }

    def _check_required(self, slots: TravelSlots, llm_missing: list) -> list:
        """
        规则引擎：检查必填项。
        与LLM的missing_slots合并，规则引擎补充LLM遗漏的硬约束。
        """
        required_fields = ["destination", "travel_days"]
        missing = []

        for field in required_fields:
            value = getattr(slots, field)
            if value is None or value == "":
                missing.append(field)

        # LLM识别的缺失项也加入（去重）
        for m in llm_missing:
            if m not in missing:
                missing.append(m)

        return missing

    def _generate_clarify_question(self, slots: TravelSlots, missing: list,
                                    feasibility: dict) -> str:
        """根据缺失项和可行性冲突，生成自然语言追问"""
        questions = []

        # 缺失项追问
        slot_names = {
            "destination": "您想去哪个城市旅行呢？",
            "travel_days": "您计划旅行几天？",
            "travel_dates": "您什么时候出发？（如7月1日）",
            "total_budget": "您的旅行预算大概是多少？",
            "travelers_count": "一共几个人出行？",
        }
        for m in missing:
            if m in slot_names:
                questions.append(slot_names[m])

        # 可行性冲突追问
        for conflict in feasibility.get("conflicts", []):
            level = conflict.get("level", "info")
            if level == "error":
                questions.append(f"⚠️ {conflict['message']}")
                questions.extend([f"  💡 {s}" for s in conflict.get("suggestions", [])])
            elif level == "warning":
                questions.append(f"ℹ️ {conflict['message']}")

        if not questions:
            return ""

        return "\\n".join(["为了更好地为您规划，请补充以下信息：", ""] + questions)
```

---

### 3.3 意图分类与情感识别 Prompt 设计

```python
# 意图分类专用Prompt（当置信度<0.7时单独调用）

INTENT_CLASSIFICATION_PROMPT = """请判断以下用户输入的意图。

意图选项：
A. new_itinerary — 用户想要一份新的旅行行程
B. modify — 用户想修改已有的行程（如"换个酒店""多加一天"）
C. query — 用户只是询问信息（如"北京有什么好玩的？""故宫门票多少？"）
D. book — 用户想要预订（如"帮我订故宫的票""订这个酒店"）
E. emergency — 用户遇到紧急情况（如"航班取消了怎么办""行李丢了"）
F. cancel — 用户取消当前操作

用户输入：{user_input}
对话历史：{messages}

请输出JSON：{{"intent": "A-F的对应值", "confidence": 0.0-1.0}}
置信度判断标准：
- 1.0：明确包含关键词（如"帮我规划行程""订酒店"）
- 0.8：语义清晰但无明确关键词
- 0.5：有歧义，需要上下文判断
- 0.2：完全不相关
"""

# 情感识别专用Prompt（影响OutputFormatAgent文案语气）

SENTIMENT_ANALYSIS_PROMPT = """请分析用户在旅行规划对话中的情感状态。

情感选项：
- excited: 兴奋期待（"好期待！""太棒了！""终于要去玩了"）
- anxious: 焦虑担忧（"会不会很累？""预算够吗？""人多不多？"）
- hesitant: 犹豫不决（"不知道去哪好""哪个更好呢？""我再想想"）
- frustrated:  frustration（"怎么这么贵！""太麻烦了""算了不去了"）
- neutral: 中性平静

用户输入：{user_input}

请输出JSON：{{"sentiment": "情感类型", "confidence": 0.0-1.0}}

注意：
- 情感会影响后续行程文案的语气（兴奋→活泼/焦虑→安抚/犹豫→推荐）
- 如果检测到frustrated且confidence>0.7，触发人工客服接管
"""
```

---

### 3.4 歧义消解策略

```python
# disambiguation.py —— 歧义消解引擎

class DisambiguationEngine:
    """
    歧义消解引擎：当用户输入信息不完整或有歧义时，
    通过追问策略澄清需求。
    """

    # 常见歧义模式库
    AMBIGUITY_PATTERNS = {
        # 目的地歧义
        "destination_vague": {
            "pattern": ["去海边", "去南方", "去有山的地方"],
            "strategy": "recommend_cities",
            "response": "您提到的目的地范围比较广。根据您的偏好，推荐以下几个城市：\\n{cities}\\n请问您更喜欢哪个？"
        },
        # 预算歧义
        "budget_vague": {
            "pattern": ["便宜点", "别太贵", "中等价位"],
            "strategy": "budget_range",
            "response": "好的，我来帮您确认预算范围。{destination}的旅行，经济型约{low}元/天，舒适型约{mid}元/天，豪华型约{high}元/天。您倾向哪种？"
        },
        # 天数歧义
        "days_vague": {
            "pattern": ["玩几天好", "多久合适", "短时间"],
            "strategy": "recommend_days",
            "response": "{destination}的经典景点大约需要{min_days}-{max_days}天可以游览完。您的时间安排是怎样的？"
        },
        # 人群歧义
        "companion_ambiguous": {
            "pattern": ["带家人", "和家人", "跟朋友"],
            "strategy": "clarify_companion",
            "response": "您提到的'家人'具体是？\\nA. 带父母（老人）\\nB. 带孩子\\nC. 夫妻/情侣\\nD. 兄弟姐妹\\n不同组合推荐的行程会有差异哦~"
        },
    }

    async def handle(self, user_input: str, slots: TravelSlots) -> Optional[str]:
        """检测歧义并返回追问文案。None表示无歧义。"""
        for name, rule in self.AMBIGUITY_PATTERNS.items():
            if any(p in user_input for p in rule["pattern"]):
                return await self._execute_strategy(rule["strategy"], slots, rule["response"])
        return None

    async def _execute_strategy(self, strategy: str, slots: TravelSlots, template: str) -> str:
        if strategy == "recommend_cities":
            # 根据兴趣推荐城市
            cities = self._recommend_cities(slots.interests)
            return template.format(cities="\\n".join([f"- {c}" for c in cities]))

        elif strategy == "budget_range":
            # 从city_info表获取预算参考
            city = slots.destination or "该城市"
            costs = await self._get_city_costs(city)
            return template.format(destination=city, low=costs["low"],
                                   mid=costs["mid"], high=costs["high"])

        elif strategy == "recommend_days":
            city = slots.destination or "该城市"
            days = self._recommend_days(city)
            return template.format(destination=city, min_days=days["min"], max_days=days["max"])

        elif strategy == "clarify_companion":
            return template

        return None

    def _recommend_cities(self, interests: list) -> list:
        """根据兴趣标签推荐城市"""
        city_map = {
            "历史": ["西安", "北京", "南京"],
            "美食": ["成都", "广州", "长沙"],
            "自然": ["丽江", "九寨沟", "张家界"],
            "海边": ["三亚", "厦门", "青岛"],
            "拍照": ["厦门", "大理", "重庆"],
        }
        cities = set()
        for interest in interests:
            for c in city_map.get(interest, []):
                cities.add(c)
        return list(cities)[:5] if cities else ["北京", "西安", "成都", "杭州", "厦门"]

    def _recommend_days(self, city: str) -> dict:
        """根据城市推荐游玩天数"""
        days_map = {
            "北京": {"min": 3, "max": 5},
            "西安": {"min": 2, "max": 4},
            "成都": {"min": 2, "max": 4},
            "上海": {"min": 2, "max": 3},
            "杭州": {"min": 2, "max": 3},
            "三亚": {"min": 3, "max": 5},
        }
        return days_map.get(city, {"min": 2, "max": 4})
```

---

### 3.5 可行性前置校验（完整规则库）

```python
# feasibility.py —— 可行性校验引擎

class FeasibilityChecker:
    """
    前置可行性校验引擎。
    在调用CP-SAT求解器前，通过规则库快速检查需求是否合理，
    避免不可行需求进入求解流程浪费算力。

    校验维度：
      1. 预算 vs 城市日均成本
      2. 人群 vs 行程长度
      3. 预约制景点提醒
      4. 孕妇安全
      5. 季节性闭园
    """

    # 城市日均成本参考（经济/标准/豪华）
    CITY_DAILY_COST = {
        "北京": {"low": 400, "mid": 700, "high": 1200},
        "上海": {"low": 450, "mid": 800, "high": 1500},
        "西安": {"low": 300, "mid": 500, "high": 900},
        "成都": {"low": 300, "mid": 550, "high": 1000},
        "杭州": {"low": 350, "mid": 600, "high": 1100},
        "重庆": {"low": 280, "mid": 500, "high": 900},
        "厦门": {"low": 350, "mid": 600, "high": 1000},
        "丽江": {"low": 250, "mid": 400, "high": 700},
        "三亚": {"low": 600, "mid": 1000, "high": 2000},
        "大理": {"low": 250, "mid": 400, "high": 700},
    }

    # 需预约景点库
    RESERVATION_REQUIRED = {
        "故宫": {"advance_days": 7, "channel": "故宫博物院官网/微信小程序",
                "note": "周一闭馆（法定节假日除外）"},
        "陕西历史博物馆": {"advance_days": 3, "channel": "陕西历史博物馆公众号",
                        "note": "周二闭馆"},
        "天安门广场": {"advance_days": 1, "channel": "天安门广场预约参观小程序"},
        "兵马俑": {"advance_days": 1, "channel": "秦始皇帝陵博物院官网"},
        "敦煌莫高窟": {"advance_days": 30, "channel": "莫高窟官网",
                      "note": "旺季需提前1个月预约"},
        "布达拉宫": {"advance_days": 7, "channel": "布达拉宫官网",
                   "note": "需提前1天确认"},
    }

    # 季节性闭园景点
    SEASONAL_CLOSURE = {
        "九寨沟": {"closed_months": [], "note": "全年开放，但冬季部分景点可能关闭"},
        "张家界": {"closed_months": [], "note": "冬季玻璃栈道可能关闭"},
        "长白山": {"closed_months": [11, 12, 1, 2, 3], "note": "11月-3月冬季封山"},
        "喀纳斯": {"closed_months": [10, 11, 12, 1, 2, 3, 4], "note": "10月-4月冬季关闭"},
    }

    def check(self, slots: TravelSlots) -> dict:
        conflicts = []

        # ── 校验1: 预算 vs 城市日均成本 ──
        self._check_budget(slots, conflicts)

        # ── 校验2: 人群 vs 行程长度 ──
        self._check_traveler_constraints(slots, conflicts)

        # ── 校验3: 预约制景点提醒 ──
        self._check_reservation(slots, conflicts)

        # ── 校验4: 孕妇安全 ──
        self._check_pregnant_safety(slots, conflicts)

        # ── 校验5: 季节性闭园 ──
        self._check_seasonal_closure(slots, conflicts)

        return {
            "feasible": not any(c["level"] == "error" for c in conflicts),
            "conflicts": conflicts,
            "conflict_count": len(conflicts),
            "error_count": sum(1 for c in conflicts if c["level"] == "error"),
        }

    def _check_budget(self, slots: TravelSlots, conflicts: list):
        dest = slots.destination
        budget = slots.total_budget
        days = slots.travel_days
        if not (dest and budget and days):
            return

        daily = budget / days
        costs = self.CITY_DAILY_COST.get(dest, {"low": 300, "mid": 500, "high": 1000})

        if daily < costs["low"] * 0.4:
            conflicts.append({
                "type": "budget_insufficient",
                "level": "error",
                "message": f"「{dest}」经济型日均预算约{costs['low']}元，"
                          f"当前日均{daily:.0f}元严重不足，可能导致无法安排住宿或交通。",
                "suggestions": [
                    f"缩短至{int(budget / costs['low'])}天（按经济型标准）",
                    "选择青旅或经济酒店",
                    "减少付费景点，多安排免费景点"
                ]
            })
        elif daily < costs["low"] * 0.7:
            conflicts.append({
                "type": "budget_tight",
                "level": "warning",
                "message": f"「{dest}」经济型日均{costs['low']}元，当前{daily:.0f}元偏紧张。",
                "suggestions": ["控制餐饮在100元/天以内", "优先选择公共交通"]
            })

    def _check_traveler_constraints(self, slots: TravelSlots, conflicts: list):
        # 老人行程长度
        if slots.has_elderly and slots.travel_days and slots.travel_days > 5:
            conflicts.append({
                "type": "elderly_too_long",
                "level": "warning",
                "message": "带老人出行建议不超过5天，长时间旅行容易疲劳。",
                "suggestions": ["缩短至3-4天", "每天安排午休时间", "减少步行类景点"]
            })

        # 儿童行程强度
        if slots.has_children and slots.pace == "intensive":
            conflicts.append({
                "type": "kid_pace_conflict",
                "level": "warning",
                "message": "带儿童不建议高强度行程，孩子容易疲劳哭闹。",
                "suggestions": ["改为moderate或relaxed节奏", "每天最多3个景点"]
            })

        # 轮椅无障碍
        if slots.has_wheelchair and not slots.destination:
            pass  # 目的地未知，后续检查
        elif slots.has_wheelchair:
            conflicts.append({
                "type": "wheelchair_note",
                "level": "info",
                "message": "已为您筛选无障碍设施完善的景点，行程中避开台阶较多的区域。",
                "suggestions": []
            })

    def _check_reservation(self, slots: TravelSlots, conflicts: list):
        for name in slots.must_visit:
            if name in self.RESERVATION_REQUIRED:
                info = self.RESERVATION_REQUIRED[name]
                conflicts.append({
                    "type": "reservation_required",
                    "level": "info",
                    "message": f"「{name}」需提前{info['advance_days']}天预约"
                              + (f"，{info['note']}" if "note" in info else ""),
                    "suggestions": [f"通过{info['channel']}预约"]
                })

    def _check_pregnant_safety(self, slots: TravelSlots, conflicts: list):
        if not slots.has_pregnant:
            return

        risky_keywords = ["徒步", "登山", "潜水", "滑雪", "漂流", "攀岩", "高原"]
        risky_interests = [k for k in slots.interests if k in risky_keywords]

        if risky_interests:
            conflicts.append({
                "type": "pregnant_safety",
                "level": "warning",
                "message": f"孕妇不建议参与「{'/'.join(risky_interests)}」类活动。",
                "suggestions": ["替换为观光类景点", "选择平坦路线", "避免高海拔地区"]
            })

    def _check_seasonal_closure(self, slots: TravelSlots, conflicts: list):
        import datetime
        dest = slots.destination
        dates_str = slots.travel_dates

        if not (dest and dates_str):
            return

        # 解析日期
        try:
            start_date = datetime.datetime.strptime(dates_str.split("~")[0], "%Y-%m-%d")
            month = start_date.month
        except:
            return

        # 检查必去景点是否季节性闭园
        for name in slots.must_visit:
            if name in self.SEASONAL_CLOSURE:
                info = self.SEASONAL_CLOSURE[name]
                if month in info.get("closed_months", []):
                    conflicts.append({
                        "type": "seasonal_closure",
                        "level": "error",
                        "message": f"「{name}」在{month}月处于闭园期，{info['note']}",
                        "suggestions": ["更换目的地", "调整出行月份"]
                    })
```


---

## 4. 用户画像层（第4层）

### 4.1 职责定位

用户画像层是系统的"记忆中枢"，解决**"这个用户是谁？之前来过吗？有什么偏好？"**的问题。

**核心任务**：
1. **身份识别**：通过 `user_id` 区分新用户 / 老用户
2. **短期记忆读取**：从 Redis 读取当前会话的上下文（最近几轮对话、当前已确认槽位）
3. **长期画像召回**：从 pgvector 读取用户的历史偏好向量（768维）+ 结构化画像
4. **画像推断槽位**：用历史画像填补当前缺失槽位（如"预算"→历史日均消费）
5. **画像更新**：行程确认后，将新偏好写回长期记忆

**关键设计**：短期记忆（Redis，TTL=30min）+ 长期画像（pgvector，持久化），两层分离。

**绝对不能做**：修改画像（只读）；调用LLM生成画像；泄露用户隐私数据给下游Agent。

---

### 4.2 数据模型：用户画像 Schema

```python
# models/user_profile.py —— 用户画像数据模型

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime

class UserProfile(BaseModel):
    """
    用户画像 —— 结构化的用户长期偏好。
    存储在 PostgreSQL user_profile 表中，向量部分存储在 pgvector 中。
    """
    # ── 基础身份 ──
    user_id: str = Field(..., description="全局唯一用户ID，如 'usr_abc123'")
    created_at: datetime = Field(..., description="首次使用时间")
    updated_at: datetime = Field(..., description="最后更新时间")
    
    # ── 人口统计学 ──
    age_group: Optional[str] = Field(None, description="年龄段: 18-25/26-35/36-50/50+")
    gender: Optional[str] = Field(None, description="性别: M/F/O/None")
    
    # ── 旅行偏好（从历史行程聚合）──
    preferred_pace: Optional[str] = Field(None, description="偏好节奏: relaxed/moderate/intensive")
    avg_trip_days: Optional[float] = Field(None, description="平均行程天数")
    avg_daily_budget: Optional[float] = Field(None, description="日均预算历史均值")
    preferred_destinations: List[str] = Field([], description="去过的城市列表")
    interest_tags: List[str] = Field([], description="兴趣标签聚合，如['历史','美食']")
    food_taboos_persistent: List[str] = Field([], description="持续性饮食禁忌")
    
    # ── 人群特征（从slots历史推断）──
    usually_travels_with: Optional[str] = Field(None, description="通常同行人类型")
    has_persistent_condition: List[str] = Field([], description="持续性身体状况: ['elderly','children','pregnant','wheelchair']")
    
    # ── 系统字段 ──
    trip_count: int = Field(0, description="完成行程次数")
    total_spent_estimate: float = Field(0.0, description="累计消费估算")
    

class TripHistory(BaseModel):
    """
    单次行程记录 —— 存储在 user_trip_history 表中。
    用于分析用户偏好变化、推荐相似行程。
    """
    trip_id: str = Field(..., description="行程唯一ID")
    user_id: str = Field(..., description="关联用户ID")
    created_at: datetime = Field(..., description="行程创建时间")
    
    # 输入参数
    destination: str
    travel_days: int
    total_budget: Optional[float]
    travelers_count: int
    travel_companion: str
    
    # 输出结果摘要
    itinerary_summary: str = Field(..., description="行程摘要JSON")
    poi_visited: List[str] = Field([], description="实际游览景点列表")
    actual_spend_estimate: Optional[float] = Field(None, description="实际花费估算")
    
    # 反馈
    user_rating: Optional[int] = Field(None, description="用户评分 1-5")
    feedback_tags: List[str] = Field([], description="反馈标签，如['walk_too_much','loved_food']")
```

---

### 4.3 数据库 DDL

```sql
-- ============================================================
-- 用户画像表 + 行程历史表 + 向量扩展
-- ============================================================

-- 启用 pgvector 扩展（仅需执行一次）
CREATE EXTENSION IF NOT EXISTS vector;

-- ── 主表：用户画像 ──
CREATE TABLE user_profile (
    user_id                 VARCHAR(32) PRIMARY KEY,    -- 如 'usr_abc123'
    
    -- 时间戳
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- 人口统计学（用户可选填）
    age_group               VARCHAR(10),                -- 18-25 / 26-35 / 36-50 / 50+
    gender                  VARCHAR(1),                 -- M / F / O
    
    -- 偏好（从行程历史自动聚合）
    preferred_pace          VARCHAR(20),                -- relaxed / moderate / intensive
    avg_trip_days           REAL,                       -- 平均行程天数
    avg_daily_budget        REAL,                       -- 日均预算均值（元）
    preferred_destinations  TEXT[],                     -- 去过的城市列表
    interest_tags           TEXT[],                     -- 兴趣标签
    food_taboos_persistent  TEXT[],                     -- 持续性饮食禁忌
    
    -- 人群
    usually_travels_with    VARCHAR(20),                -- solo / couple / family_kid / family_elder / friends
    has_persistent_condition TEXT[],                    -- elderly / children / pregnant / wheelchair
    
    -- 统计
    trip_count              INT NOT NULL DEFAULT 0,
    total_spent_estimate    REAL NOT NULL DEFAULT 0.0,
    
    -- 隐私：加密敏感字段（预留）
    encrypted_phone         BYTEA,                      -- AES-256-GCM 加密
    encrypted_email         BYTEA,
    
    -- 向量（偏好嵌入，768维）
    preference_vector       VECTOR(768),                -- bge-large-zh-v1.5 产出
    
    -- 索引
    CONSTRAINT chk_age_group CHECK (age_group IN ('18-25','26-35','36-50','50+')),
    CONSTRAINT chk_pace CHECK (preferred_pace IN ('relaxed','moderate','intensive'))
);

-- 注释
COMMENT ON TABLE user_profile IS '用户画像主表：长期偏好+向量表示';
COMMENT ON COLUMN user_profile.preference_vector IS 'bge-large-zh-v1.5 768维偏好嵌入向量';

-- ── 辅助表：行程历史 ──
CREATE TABLE user_trip_history (
    trip_id                 VARCHAR(32) PRIMARY KEY,
    user_id                 VARCHAR(32) NOT NULL REFERENCES user_profile(user_id) ON DELETE CASCADE,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    destination             VARCHAR(50) NOT NULL,
    travel_days             INT NOT NULL,
    total_budget            REAL,
    travelers_count         INT NOT NULL DEFAULT 1,
    travel_companion        VARCHAR(20) NOT NULL,
    
    -- 输出摘要（JSONB存储完整行程概要）
    itinerary_summary       JSONB,
    poi_visited             TEXT[],
    actual_spend_estimate   REAL,
    
    -- 用户反馈
    user_rating             INT CHECK (user_rating BETWEEN 1 AND 5),
    feedback_tags           TEXT[],
    
    -- 用于相似行程推荐
    trip_vector             VECTOR(768)                 -- 行程描述嵌入
);

COMMENT ON TABLE user_trip_history IS '用户行程历史：用于偏好分析和相似推荐';

-- ── 索引 ──
-- 向量相似度检索索引（HNSW，推荐参数）
CREATE INDEX idx_user_profile_vector ON user_profile 
    USING hnsw (preference_vector vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_trip_history_vector ON user_trip_history 
    USING hnsw (trip_vector vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- 常用查询索引
CREATE INDEX idx_trip_history_user ON user_trip_history (user_id, created_at DESC);
CREATE INDEX idx_profile_updated ON user_profile (updated_at DESC);

-- ── 触发器：自动更新 updated_at ──
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_user_profile_updated
    BEFORE UPDATE ON user_profile
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
```

---

### 4.4 UserProfileRecallAgent 完整实现

```python
# agents/profile_recall.py —— 用户画像召回Agent

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json
import numpy as np

class UserProfileRecallAgent:
    """
    用户画像召回Agent：短期Redis + 长期pgvector 双层记忆系统。
    
    处理流程：
      1. 检查 user_id，区分新老用户
      2. 读Redis短期记忆（当前会话上下文）
      3. 读pgvector长期画像（历史偏好向量）
      4. 画像推断：用历史数据填补当前缺失槽位
      5. 返回 merged_profile + inferred_slots + is_new_user
    
    约束：只读不写（画像更新由Confirm后的独立流程触发）。
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Redis连接（短连接池）
        import redis.asyncio as redis
        self.redis = redis.Redis(
            host=config.get("REDIS_HOST", "localhost"),
            port=config.get("REDIS_PORT", 6379),
            db=config.get("REDIS_DB", 0),
            decode_responses=True,
            socket_connect_timeout=2,   # 快速失败
            socket_timeout=2,
            max_connections=20,
        )
        # PostgreSQL + pgvector
        import asyncpg
        self.pg_pool = None  # 懒加载，由外部传入
        # Embedding模型
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer("BAAI/bge-large-zh-v1.5")

    async def recall(self, user_id: str, slots: Dict[str, Any],
                     messages: List[Dict]) -> Dict[str, Any]:
        """
        主入口：召回用户画像。
        
        返回：{
            "user_profile": dict or {},
            "preference_vector": list or None,
            "inferred_slots": dict,
            "is_new_user": bool,
        }
        """
        # 1. 匿名用户直接返回空画像
        if not user_id or user_id == "anonymous":
            return {
                "user_profile": {},
                "preference_vector": None,
                "inferred_slots": {},
                "is_new_user": True,
            }

        # 2. 读取两层记忆
        short_term = await self._read_short_term(user_id)
        long_term = await self._read_long_term(user_id)

        # 3. 判断是否新用户
        is_new = long_term.get("is_new_user", True)

        # 4. 合并画像（长期优先，短期补充）
        merged_profile = self._merge_profiles(short_term, long_term)

        # 5. 用画像推断缺失槽位
        current_slots = slots or {}
        inferred = self._infer_slots(merged_profile, current_slots)

        # 6. 获取偏好向量
        pref_vector = long_term.get("preference_vector") or short_term.get("preference_vector")

        return {
            "user_profile": merged_profile,
            "preference_vector": pref_vector,
            "inferred_slots": inferred,
            "is_new_user": is_new,
        }

    # ═══════════════════════════════════════════
    # 短期记忆：Redis 读写
    # ═══════════════════════════════════════════

    async def _read_short_term(self, user_id: str) -> Dict[str, Any]:
        """
        读取Redis短期记忆。
        Key格式: travel_agent:session:{user_id}:profile
        TTL: 30分钟（与SessionManager一致）
        """
        key = f"travel_agent:session:{user_id}:profile"
        try:
            data = await self.redis.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            # Redis失败不阻塞，返回空
            pass
        return {}

    async def _write_short_term(self, user_id: str, profile: Dict[str, Any],
                                ttl_minutes: int = 30):
        """
        写入Redis短期记忆。
        由SessionManager在槽位确认后调用，不是本Agent职责。
        这里提供方法供上游调用。
        """
        key = f"travel_agent:session:{user_id}:profile"
        try:
            await self.redis.setex(
                key,
                timedelta(minutes=ttl_minutes),
                json.dumps(profile, default=str)
            )
        except Exception:
            pass  # 短期记忆失败不阻塞主流程

    # ═══════════════════════════════════════════
    # 长期记忆：PostgreSQL + pgvector 读写
    # ═══════════════════════════════════════════

    async def _read_long_term(self, user_id: str) -> Dict[str, Any]:
        """
        读取PostgreSQL长期画像。
        包含：结构化字段 + preference_vector（768维）。
        """
        if not self.pg_pool:
            # 懒初始化连接池
            import asyncpg
            self.pg_pool = await asyncpg.create_pool(
                self.config.get("DATABASE_URL", "postgresql://user:pass@localhost/travel_agent"),
                min_size=2, max_size=10,
                command_timeout=5,
            )

        try:
            async with self.pg_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT user_id, created_at, updated_at,
                           age_group, gender,
                           preferred_pace, avg_trip_days, avg_daily_budget,
                           preferred_destinations, interest_tags,
                           food_taboos_persistent,
                           usually_travels_with, has_persistent_condition,
                           trip_count, total_spent_estimate,
                           preference_vector::text as vector_str
                    FROM user_profile
                    WHERE user_id = $1
                    """,
                    user_id
                )

                if not row:
                    return {"is_new_user": True}

                # 解析向量（从字符串转为float列表）
                vector = None
                if row["vector_str"]:
                    # pgvector 返回格式: "[0.1, 0.2, ...]"
                    vector_str = row["vector_str"].strip("[]")
                    vector = [float(x) for x in vector_str.split(",") if x]

                return {
                    "is_new_user": False,
                    "user_id": row["user_id"],
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                    "age_group": row["age_group"],
                    "gender": row["gender"],
                    "preferred_pace": row["preferred_pace"],
                    "avg_trip_days": row["avg_trip_days"],
                    "avg_daily_budget": row["avg_daily_budget"],
                    "preferred_destinations": row["preferred_destinations"] or [],
                    "interest_tags": row["interest_tags"] or [],
                    "food_taboos_persistent": row["food_taboos_persistent"] or [],
                    "usually_travels_with": row["usually_travels_with"],
                    "has_persistent_condition": row["has_persistent_condition"] or [],
                    "trip_count": row["trip_count"],
                    "total_spent_estimate": row["total_spent_estimate"],
                    "preference_vector": vector,
                }

        except Exception as e:
            # DB失败返回空画像，不阻塞流程
            return {"is_new_user": True, "error": str(e)[:100]}

    async def find_similar_users(self, user_id: str, top_k: int = 5) -> List[Dict]:
        """
        向量相似用户查找（用于协同过滤推荐）。
        排除自己，按余弦相似度排序。
        """
        if not self.pg_pool:
            return []

        try:
            async with self.pg_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT 
                        up.user_id,
                        up.interest_tags,
                        up.preferred_destinations,
                        up.preference_vector <=> (
                            SELECT preference_vector FROM user_profile WHERE user_id = $1
                        ) as distance
                    FROM user_profile up
                    WHERE up.user_id != $1
                      AND up.preference_vector IS NOT NULL
                    ORDER BY distance ASC
                    LIMIT $2
                    """,
                    user_id, top_k
                )
                return [{
                    "user_id": r["user_id"],
                    "interest_tags": r["interest_tags"],
                    "preferred_destinations": r["preferred_destinations"],
                    "cosine_similarity": 1.0 - float(r["distance"]),
                } for r in rows]
        except Exception:
            return []

    async def find_similar_trips(self, user_id: str, destination: str,
                                  top_k: int = 3) -> List[Dict]:
        """
        查找该用户去某城市的相似历史行程。
        用于推荐"上次去北京的行程"作为参考。
        """
        if not self.pg_pool:
            return []

        try:
            async with self.pg_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT trip_id, destination, travel_days, total_budget,
                           itinerary_summary, poi_visited, user_rating,
                           created_at
                    FROM user_trip_history
                    WHERE user_id = $1 AND destination = $2
                    ORDER BY created_at DESC
                    LIMIT $3
                    """,
                    user_id, destination, top_k
                )
                return [{
                    "trip_id": r["trip_id"],
                    "destination": r["destination"],
                    "travel_days": r["travel_days"],
                    "total_budget": r["total_budget"],
                    "poi_visited": r["poi_visited"],
                    "user_rating": r["user_rating"],
                    "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                } for r in rows]
        except Exception:
            return []

    # ═══════════════════════════════════════════
    # 画像合并与槽位推断
    # ═══════════════════════════════════════════

    def _merge_profiles(self, short: Dict, long: Dict) -> Dict[str, Any]:
        """
        合并短期+长期画像。
        策略：长期优先（更稳定），短期补充（本次会话的新信息）。
        """
        merged = dict(long)  # 以长期为基础
        merged.pop("is_new_user", None)
        merged.pop("error", None)
        merged.pop("preference_vector", None)

        # 短期记忆覆盖（短期信息更新鲜）
        for key in ["age_group", "gender", "preferred_pace", 
                    "usually_travels_with", "has_persistent_condition"]:
            if key in short and short[key] is not None:
                merged[key] = short[key]

        # 兴趣标签：合并去重
        short_tags = set(short.get("interest_tags", []))
        long_tags = set(long.get("interest_tags", []))
        merged["interest_tags"] = list(short_tags | long_tags)

        # 饮食禁忌：合并去重
        short_taboos = set(short.get("food_taboos_persistent", []))
        long_taboos = set(long.get("food_taboos_persistent", []))
        merged["food_taboos_persistent"] = list(short_taboos | long_taboos)

        return merged

    def _infer_slots(self, profile: Dict[str, Any],
                     current_slots: Dict[str, Any]) -> Dict[str, Any]:
        """
        用画像推断当前缺失槽位。
        
        推断规则（按置信度排序）：
          1. 预算 → 历史日均预算 × 天数
          2. 节奏 → 历史偏好节奏
          3. 人群 → 历史通常同行人
          4. 兴趣 → 历史兴趣标签
          5. 饮食禁忌 → 历史持续性禁忌
          6. 目的地 → 相似用户常去的目的地
        
        返回：{slot_name: inferred_value}，只包含有把握推断的项。
        """
        inferred = {}
        if not profile:
            return inferred

        # 推断1：总预算（最高价值推断）
        if not current_slots.get("total_budget"):
            avg_daily = profile.get("avg_daily_budget")
            days = current_slots.get("travel_days")
            if avg_daily and days:
                inferred["total_budget"] = round(avg_daily * days, 2)
                inferred["_inferred_from"] = "avg_daily_budget"
                inferred["_inference_confidence"] = 0.7  # 日均×天数有一定波动

        # 推断2：节奏
        if not current_slots.get("pace"):
            pref_pace = profile.get("preferred_pace")
            if pref_pace:
                inferred["pace"] = pref_pace
                inferred["_inferred_from"] = "preferred_pace"
                inferred["_inference_confidence"] = 0.8

        # 推断3：人群/同行人类型
        if not current_slots.get("travel_companion"):
            usually_with = profile.get("usually_travels_with")
            if usually_with:
                inferred["travel_companion"] = usually_with

        # 推断4：兴趣标签（补充而非覆盖）
        if not current_slots.get("interests"):
            hist_tags = profile.get("interest_tags", [])
            if hist_tags:
                inferred["interests"] = hist_tags[:5]  # 最多取5个

        # 推断5：持续性饮食禁忌（重要！健康相关）
        hist_taboos = profile.get("food_taboos_persistent", [])
        current_taboos = current_slots.get("food_taboos", [])
        if hist_taboos:
            # 合并历史禁忌到当前（确保不遗漏过敏等关键信息）
            merged_taboos = list(set(current_taboos + hist_taboos))
            if len(merged_taboos) > len(current_taboos):
                inferred["food_taboos"] = merged_taboos
                inferred["_inferred_from"] = "food_taboos_persistent"
                inferred["_inference_note"] = "从历史行程合并持续性饮食禁忌"

        # 推断6：身体状况（持续性条件自动继承）
        persistent = profile.get("has_persistent_condition", [])
        if "elderly" in persistent and not current_slots.get("has_elderly"):
            inferred["has_elderly"] = True
        if "children" in persistent and not current_slots.get("has_children"):
            inferred["has_children"] = True
        if "pregnant" in persistent and not current_slots.get("has_pregnant"):
            inferred["has_pregnant"] = True
        if "wheelchair" in persistent and not current_slots.get("has_wheelchair"):
            inferred["has_wheelchair"] = True

        return inferred

    # ═══════════════════════════════════════════
    # 画像更新（行程确认后调用）
    # ═══════════════════════════════════════════

    async def update_profile_after_confirm(self, user_id: str,
                                           confirmed_slots: Dict[str, Any],
                                           itinerary: List[Dict]) -> bool:
        """
        用户确认行程后，更新长期画像。
        
        更新内容：
          1. 添加到历史行程
          2. 重新计算平均天数/日均预算
          3. 更新兴趣标签
          4. 重新生成偏好向量
        
        注意：这不是Agent主流程的一部分，由确认回调触发。
        """
        if not user_id or user_id == "anonymous":
            return False

        try:
            async with self.pg_pool.acquire() as conn:
                # 1. 插入行程历史
                trip_id = f"trip_{datetime.now().strftime('%Y%m%d%H%M%S')}_{user_id[:8]}"
                poi_list = []
                for day in itinerary:
                    for act in day.get("schedule", []):
                        poi_list.append(act.get("spot_name", ""))

                # 生成行程向量（用于相似行程推荐）
                trip_desc = f"去{confirmed_slots.get('destination','')}"
                trip_desc += f"{confirmed_slots.get('travel_days','')}天"
                trip_desc += f"，游览{','.join(poi_list[:10])}"
                trip_vector = self.embedder.encode(trip_desc).tolist()

                await conn.execute(
                    """
                    INSERT INTO user_trip_history 
                        (trip_id, user_id, destination, travel_days, total_budget,
                         travelers_count, travel_companion, poi_visited, trip_vector)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,
                    trip_id, user_id,
                    confirmed_slots.get("destination", ""),
                    confirmed_slots.get("travel_days", 0),
                    confirmed_slots.get("total_budget"),
                    confirmed_slots.get("travelers_count", 1),
                    confirmed_slots.get("travel_companion", "solo"),
                    poi_list,
                    json.dumps(trip_vector)
                )

                # 2. 更新画像统计
                await conn.execute(
                    """
                    UPDATE user_profile
                    SET 
                        trip_count = trip_count + 1,
                        preferred_destinations = array_append_unique(
                            COALESCE(preferred_destinations, ARRAY[]::text[]), 
                            $2
                        ),
                        -- 重新计算平均值
                        avg_trip_days = (
                            SELECT AVG(travel_days) FROM user_trip_history 
                            WHERE user_id = $1
                        ),
                        avg_daily_budget = (
                            SELECT AVG(total_budget / NULLIF(travel_days, 0)) 
                            FROM user_trip_history 
                            WHERE user_id = $1 AND total_budget IS NOT NULL
                        ),
                        -- 更新兴趣标签（合并本次的）
                        interest_tags = array_cat_unique(
                            COALESCE(interest_tags, ARRAY[]::text[]),
                            $3::text[]
                        ),
                        -- 更新持续性饮食禁忌
                        food_taboos_persistent = array_cat_unique(
                            COALESCE(food_taboos_persistent, ARRAY[]::text[]),
                            $4::text[]
                        ),
                        -- 更新人群偏好
                        usually_travels_with = COALESCE($5, usually_travels_with),
                        -- 更新条件
                        has_persistent_condition = array_cat_unique(
                            COALESCE(has_persistent_condition, ARRAY[]::text[]),
                            $6::text[]
                        )
                    WHERE user_id = $1
                    """,
                    user_id,
                    confirmed_slots.get("destination", ""),
                    confirmed_slots.get("interests", []),
                    confirmed_slots.get("food_taboos", []),
                    confirmed_slots.get("travel_companion"),
                    self._extract_conditions(confirmed_slots)
                )

                # 3. 重新生成偏好向量
                await self._regenerate_preference_vector(user_id)

                return True

        except Exception as e:
            # 画像更新失败不影响主流程
            return False

    async def _regenerate_preference_vector(self, user_id: str):
        """用最新的画像文本重新生成768维偏好向量。"""
        async with self.pg_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT interest_tags, preferred_destinations, 
                       preferred_pace, usually_travels_with
                FROM user_profile WHERE user_id = $1
                """,
                user_id
            )
            if not row:
                return

            # 构建画像描述文本
            parts = []
            if row["interest_tags"]:
                parts.append(f"喜欢{','.join(row['interest_tags'][:8])}")
            if row["preferred_destinations"]:
                parts.append(f"去过{','.join(row['preferred_destinations'][:5])}")
            if row["preferred_pace"]:
                parts.append(f"偏好{row['preferred_pace']}节奏")
            if row["usually_travels_with"]:
                parts.append(f"通常{row['usually_travels_with']}出行")

            profile_text = "；".join(parts) if parts else "旅行爱好者"
            new_vector = self.embedder.encode(profile_text).tolist()

            # 更新向量
            await conn.execute(
                """
                UPDATE user_profile 
                SET preference_vector = $2::vector
                WHERE user_id = $1
                """,
                user_id, json.dumps(new_vector)
            )

    @staticmethod
    def _extract_conditions(slots: Dict) -> List[str]:
        """从槽位中提取持续性身体状况条件。"""
        conditions = []
        if slots.get("has_elderly"): conditions.append("elderly")
        if slots.get("has_children"): conditions.append("children")
        if slots.get("has_pregnant"): conditions.append("pregnant")
        if slots.get("has_wheelchair"): conditions.append("wheelchair")
        return conditions

    # ═══════════════════════════════════════════
    # 向量工具函数
    # ═══════════════════════════════════════════

    def compute_profile_embedding(self, profile_text: str) -> List[float]:
        """
        将画像描述文本转为768维向量。
        用于新用户画像初始化或调试。
        """
        return self.embedder.encode(profile_text).tolist()

    async def close(self):
        """清理资源。"""
        await self.redis.close()
        if self.pg_pool:
            await self.pg_pool.close()
```

---

### 4.5 隐私隔离方案

```python
# privacy.py —— 隐私保护模块

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os
import hashlib

class PrivacyGuard:
    """
    隐私保护中心：负责敏感数据的加密存储和访问控制。
    
    保护原则：
      1. 最小采集：只收集行程规划必需的信息
      2. 透明告知：首次使用时明确告知数据用途
      3. 用户控制：用户可随时导出/删除自己的数据
      4. 加密存储：AES-256-GCM 加密PII字段
      5. 脱敏日志：所有日志中用户ID做哈希处理
    
    加密范围：phone、email、身份证号（如有）。
    不加密：user_id（需用于关联）、兴趣标签、行程数据（业务必需）。
    """

    def __init__(self, master_key: bytes = None):
        """
        master_key: 32字节AES密钥，从环境变量 TRAVEL_AGENT_MASTER_KEY 读取。
        """
        if master_key is None:
            key_hex = os.environ.get("TRAVEL_AGENT_MASTER_KEY")
            if not key_hex:
                raise ValueError("TRAVEL_AGENT_MASTER_KEY 环境变量未设置")
            master_key = bytes.fromhex(key_hex)
        
        if len(master_key) != 32:
            raise ValueError("Master key must be 32 bytes (256 bits)")
        
        self.aesgcm = AESGCM(master_key)

    def encrypt_pii(self, plaintext: str, associated_data: bytes = None) -> bytes:
        """
        加密PII数据。返回: nonce(12B) + ciphertext + tag(16B)。
        """
        if not plaintext:
            return None
        nonce = os.urandom(12)  # 每次随机IV
        data = plaintext.encode("utf-8")
        encrypted = self.aesgcm.encrypt(nonce, data, associated_data)
        return nonce + encrypted  # 前缀携带nonce

    def decrypt_pii(self, ciphertext: bytes, associated_data: bytes = None) -> str:
        """解密PII数据。"""
        if not ciphertext:
            return None
        nonce = ciphertext[:12]
        encrypted = ciphertext[12:]
        decrypted = self.aesgcm.decrypt(nonce, encrypted, associated_data)
        return decrypted.decode("utf-8")

    @staticmethod
    def hash_user_id_for_logs(user_id: str) -> str:
        """
        日志中使用的脱敏用户ID。
        不可逆哈希，保护真实身份。
        """
        return hashlib.sha256(f"travel_agent:{user_id}".encode()).hexdigest()[:16]

    @staticmethod
    def anonymize_itinerary_for_logs(itinerary: list) -> list:
        """
        行程脱敏：移除所有可识别个人身份的信息，保留结构用于调试。
        """
        if not itinerary:
            return []
        return [
            {
                "day": day.get("day"),
                "poi_count": len(day.get("schedule", [])),
                "total_minutes": sum(a.get("play_minute", 0) 
                                     for a in day.get("schedule", [])),
            }
            for day in itinerary
        ]

    def export_user_data(self, user_id: str, pg_pool) -> dict:
        """
        GDPR数据导出：返回用户的所有数据，JSON格式。
        用户有权获得其数据的完整副本。
        """
        # 由独立接口实现，这里定义契约
        return {
            "user_id": user_id,
            "export_version": "1.0",
            "profile": {},      # SELECT * FROM user_profile WHERE user_id=
            "trip_history": [], # SELECT * FROM user_trip_history WHERE user_id=
            "sessions": [],     # Redis中最近30天的会话
        }

    async def delete_user_data(self, user_id: str, pg_pool) -> bool:
        """
        GDPR完全删除（Right to be Forgotten）。
        删除PostgreSQL + Redis 中的所有用户数据。
        """
        try:
            # 删除PostgreSQL数据
            async with pg_pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM user_trip_history WHERE user_id = $1",
                    user_id
                )
                await conn.execute(
                    "DELETE FROM user_profile WHERE user_id = $1",
                    user_id
                )
            
            # 删除Redis数据
            import redis.asyncio as redis
            r = redis.Redis(decode_responses=True)
            keys = await r.keys(f"travel_agent:*:{user_id}:*")
            if keys:
                await r.delete(*keys)
            
            return True
        except Exception:
            return False


# 使用示例
PRIVACY_NOTICE = """
【隐私说明】
为了给您提供更精准的行程规划，我们需要收集以下信息：

1. 行程信息：目的地、天数、预算、同行人（必需）
2. 偏好信息：兴趣标签、饮食偏好（用于个性化推荐）
3. 历史行程：用于学习您的旅行偏好

我们承诺：
- 不会将您的数据出售给第三方
- 敏感信息（手机号、邮箱）会加密存储
- 您可以随时导出或删除自己的数据
- 如需人工客服，对话内容可能会被审查

输入"同意"即表示您理解并接受上述条款。
"""
```

---

### 4.6 记忆冲突解决策略

```python
# conflict_resolver.py —— 记忆冲突解决器

class MemoryConflictResolver:
    """
    当短期记忆（本次会话）与长期画像（历史数据）发生冲突时的解决策略。
    
    冲突场景：
      1. 用户本次说"我不吃辣"，但历史记录中有"辣"
      2. 用户本次带老人，但历史通常 solo 出行
      3. 用户本次预算5000，历史日均200
    
    解决原则：短期记忆优先（用户最新表达为准），但记录冲突供后续分析。
    """

    @staticmethod
    def resolve(profile: dict, current_slots: dict) -> dict:
        """
        检测并解决短期与长期记忆的冲突。
        
        返回：{
            "resolved_profile": dict,   # 冲突解决后的画像
            "conflicts": [{             # 检测到的冲突列表
                "field": str,
                "long_term_value": any,
                "short_term_value": any,
                "resolution": "short_wins|long_wins|merged",
                "note": str
            }]
        }
        """
        conflicts = []
        resolved = dict(profile)

        # 冲突1：饮食禁忌（短期绝对优先，安全相关）
        short_taboos = set(current_slots.get("food_taboos", []))
        long_taboos = set(profile.get("food_taboos_persistent", []))
        
        if short_taboos != long_taboos:
            # 短期添加了新的禁忌 → 短期优先
            if short_taboos - long_taboos:
                conflicts.append({
                    "field": "food_taboos",
                    "long_term_value": list(long_taboos),
                    "short_term_value": list(short_taboos),
                    "resolution": "short_wins",
                    "note": "用户在本次会话中更新了饮食禁忌，以最新为准"
                })
                resolved["food_taboos_persistent"] = list(short_taboos)

        # 冲突2：人群类型
        short_companion = current_slots.get("travel_companion")
        long_companion = profile.get("usually_travels_with")
        
        if short_companion and long_companion and short_companion != long_companion:
            conflicts.append({
                "field": "travel_companion",
                "long_term_value": long_companion,
                "short_term_value": short_companion,
                "resolution": "short_wins",
                "note": "本次出行类型与历史不同，以本次为准，但保留历史记录"
            })
            resolved["usually_travels_with"] = short_companion

        # 冲突3：身体状况
        conditions_map = {
            "has_elderly": "elderly",
            "has_children": "children",
            "has_pregnant": "pregnant",
            "has_wheelchair": "wheelchair",
        }
        current_conditions = set()
        for slot_key, cond in conditions_map.items():
            if current_slots.get(slot_key):
                current_conditions.add(cond)
        
        long_conditions = set(profile.get("has_persistent_condition", []))
        
        if current_conditions != long_conditions:
            new_conditions = current_conditions - long_conditions
            removed_conditions = long_conditions - current_conditions
            
            if new_conditions:
                conflicts.append({
                    "field": "has_persistent_condition",
                    "long_term_value": list(long_conditions),
                    "short_term_value": list(current_conditions),
                    "resolution": "merged",
                    "note": f"新增持续性条件: {new_conditions}。注意：{removed_conditions} 本次未提及，但历史存在。"
                })
                # 合并：新增的加入，移除的保留（保守策略，安全相关）
                resolved["has_persistent_condition"] = list(current_conditions | long_conditions)

        return {
            "resolved_profile": resolved,
            "conflicts": conflicts,
        }

    @staticmethod
    def should_update_long_term(conflicts: list) -> bool:
        """
        判断冲突解决后，是否应该立即更新长期画像。
        
        策略：
        - 饮食禁忌变更 → 是（安全相关，立即持久化）
        - 人群变更 → 否（可能是临时出行，需多次确认）
        - 身体状况变更 → 是（健康相关，立即持久化）
        
        不需要每次都写DB，只在确认行程后批量更新。
        这里返回flags，由上游决定是否更新。
        """
        flags = {"immediate_update": False, "reasons": []}
        
        for c in conflicts:
            if c["field"] == "food_taboos":
                flags["immediate_update"] = True
                flags["reasons"].append("饮食禁忌变更需立即持久化")
            elif c["field"] == "has_persistent_condition":
                flags["immediate_update"] = True
                flags["reasons"].append("身体状况变更需立即持久化")
        
        return flags
```

---

### 4.7 画像层输出数据契约

```python
# UserProfileRecallAgent → TravelRetrievalRAGAgent 的数据契约

profile_to_retrieve_contract = {
    # 输入（来自上游）
    "slots": TravelSlots,           # 当前已确认的槽位（第3层产出）
    
    # 输出（给下游）
    "user_profile": {
        "user_id": str,
        "age_group": str or None,
        "preferred_pace": str or None,
        "avg_trip_days": float or None,
        "avg_daily_budget": float or None,
        "preferred_destinations": [str],
        "interest_tags": [str],
        "food_taboos_persistent": [str],
        "usually_travels_with": str or None,
        "has_persistent_condition": [str],
        "trip_count": int,
    },
    "preference_vector": [float] * 768 or None,  # 768维向量
    "inferred_slots": {
        # 只包含成功推断的槽位
        # "total_budget": 3000.0,
        # "pace": "relaxed",
        # "food_taboos": ["不吃辣", "海鲜过敏"],
        # "_inferred_from": "avg_daily_budget",
        # "_inference_confidence": 0.7,
    },
    "is_new_user": bool,
    "conflicts_resolved": [          # 记忆冲突解决记录
        {"field": str, "resolution": str, "note": str}
    ] or [],
}
```

**与下游的衔接**：画像层完成后，`slots` + `inferred_slots` + `user_profile` 一并传给第5层（TravelRetrievalRAGAgent），用于个性化POI检索。画像向量（preference_vector）可用于相似用户推荐（P2+阶段）。

---

### 4.8 性能指标与监控

| 指标 | 目标值 | 说明 |
|------|--------|------|
| Redis读取延迟 | <5ms | 短期记忆读取 |
| pgvector向量检索 | <20ms | HNSW索引，768维 |
| 全画像召回耗时 | <50ms | P99，含合并+推断 |
| 向量生成耗时 | <100ms | bge-large-zh-v1.5，CPU |
| 画像更新耗时 | <200ms | 行程确认后异步更新 |
| 隐私加密耗时 | <1ms | AES-256-GCM |
| Redis命中率 | >95% | 短期记忆命中率 |
| pgvector查询成功率 | >99.9% | 向量库可用性 |

---

## 上半部分完结

### 覆盖范围总结

| 章节 | 内容 | 代码量 | 核心产出 |
|------|------|--------|---------|
| 第0章 | 8层架构全景 + 技术选型 | ~50行 | 架构图 + 选型表 |
| 第0.5章 | 6核心+3增值Agent定义 | ~80行 | Agent职责矩阵 |
| 第1章 | LangGraph编排调度 | ~600行 | AgentState + 6Node + 5路由 + Graph构建 |
| 第2章 | 感知层4类输入 | ~350行 | SSE + PaddleOCR + PyMuPDF + Webhook |
| 第3章 | 意图理解层 | ~450行 | TravelSlots + Prompt + 歧义消解 + 可行性校验 |
| 第4章 | 用户画像层 | ~500行 | ProfileAgent + Redis/pgvector + DDL + 隐私 + 冲突解决 |

**总计**：~2,030行代码 + 注释，覆盖"用户进来 → 系统理解 → 召回画像 → 输出结构化需求"完整链路。

### 与下半部分的衔接点

上半部分的最终产出是 **`AgentState.slots`**（完整TravelSlots）+ **`AgentState.user_profile`**（画像）+ **`AgentState.inferred_slots`**（推断槽位）。这三个字段作为输入传递给下半部分的第5层（TravelRetrievalRAGAgent），触发POI混合检索流程。

---

> 下半部分覆盖：第5层（知识库RAG检索）→ 第6层（CP-SAT v4.0规划引擎）→ 第7层（工具调用）→ 第8层（交互输出）+ 第9章（监控）+ 第10章（部署）+ 第11章（Roadmap）。
