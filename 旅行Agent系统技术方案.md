# 旅行 Agent 系统 - 详细技术方案

> **版本**：v1.0
> **目标**：可直接指导 AI 逐步实现的完整技术方案
> **范围**：MVP Phase 1（单轮单城市行程生成）
> **预计开发周期**：3 周

---

## 目录

1. [技术栈与环境](#一技术栈与环境)
2. [项目初始化](#二项目初始化)
3. [数据库设计](#三数据库设计)
4. [后端技术方案](#四后端技术方案)
5. [前端技术方案](#五前端技术方案)
6. [实现路线图（Step by Step）](#六实现路线图step-by-step)
7. [API 接口规范](#七api-接口规范)
8. [部署方案](#八部署方案)

---

## 一、技术栈与环境

### 1.1 后端技术栈

| 层级 | 技术 | 版本 | 用途 |
|------|------|------|------|
| 语言 | Python | 3.11+ | 主开发语言 |
| Web 框架 | FastAPI | 0.110+ | REST API + WebSocket |
| 异步 | asyncio | 内置 | Agent 并行执行 |
| LLM SDK | openai | 1.0+ | LLM 调用（兼容 Claude/通义/DeepSeek） |
| 数据验证 | Pydantic | 2.0+ | 数据模型 + API Schema |
| 数据库 | SQLAlchemy + asyncpg | 2.0+ | ORM + PostgreSQL 异步驱动 |
| 缓存 | redis-py | 5.0+ | Redis 异步客户端 |
| 爬虫 | aiohttp | 3.9+ | 异步 HTTP 请求 |
| HTML 解析 | beautifulsoup4 | 4.12+ | 网页内容提取 |
| 搜索 | duckduckgo-search | 4.0+ | 搜索引擎 API |
| 科学计算 | numpy + scikit-learn | 1.26+ / 1.4+ | 聚类算法（DBSCAN） |
| 测试 | pytest + pytest-asyncio | 8.0+ | 单元测试 + 异步测试 |

### 1.2 前端技术栈

| 层级 | 技术 | 版本 | 用途 |
|------|------|------|------|
| 框架 | Next.js | 14+ | React 全栈框架 |
| UI 库 | React | 18+ | UI 组件 |
| 样式 | Tailwind CSS | 3.4+ | 原子化 CSS |
| 组件库 | shadcn/ui | latest | 基础 UI 组件 |
| 状态管理 | Zustand | 4.5+ | 全局状态 |
| HTTP 客户端 | axios | 1.6+ | API 请求 |
| 图标 | lucide-react | 0.400+ | 图标库 |

### 1.3 基础设施

| 服务 | 技术 | 用途 |
|------|------|------|
| 数据库 | PostgreSQL 16 | 持久化存储 |
| 缓存 | Redis 7 | Session 状态 + 缓存 |
| 部署 | Docker + Docker Compose | 容器化部署 |

### 1.4 LLM 配置

```yaml
# 推荐使用 Claude Sonnet 4（通过 OpenAI 兼容接口）
llm:
  provider: openai_compatible  # 支持 Claude/通义/DeepSeek
  base_url: "https://api.anthropic.com/v1"  # 或其他兼容接口
  model: "claude-sonnet-4-6"
  temperature: 0.3  # 意图识别用低温度
  max_tokens: 4096
  
  # 结构化输出专用（高温度用于方案生成）
  proposal_generation:
    temperature: 0.7
    max_tokens: 8192
```

---

## 二、项目初始化

### 2.1 目录结构创建

```bash
# 创建项目根目录
mkdir travel-agent && cd travel-agent

# 后端目录
mkdir -p backend/src/{agents,core,skills,planner,memory,rag,crawler,data,api/{routes,middleware},config}
mkdir -p backend/tests/{unit,integration}

# 前端目录（使用 Next.js App Router）
npx create-next-app@latest frontend --typescript --tailwind --eslint --app --src-dir --import-alias "@/*"

# 前端额外目录
cd frontend
mkdir -p src/components/{layout,chat,itinerary,panels,sidebar}
mkdir -p src/hooks src/stores src/types src/lib
```

### 2.2 后端依赖配置

**`backend/pyproject.toml`**

```toml
[project]
name = "travel-agent"
version = "0.1.0"
description = "AI-driven travel planning agent"
requires-python = ">=3.11"
dependencies = [
    # Web 框架
    "fastapi>=0.110.0",
    "uvicorn[standard]>=0.27.0",
    "python-socketio>=5.11.0",
    "websockets>=12.0",
    
    # 数据验证
    "pydantic>=2.6.0",
    "pydantic-settings>=2.1.0",
    
    # 数据库
    "sqlalchemy[asyncio]>=2.0.0",
    "asyncpg>=0.29.0",
    "alembic>=1.13.0",
    
    # 缓存
    "redis>=5.0.0",
    
    # LLM
    "openai>=1.12.0",
    "httpx>=0.27.0",
    
    # 爬虫
    "aiohttp>=3.9.0",
    "beautifulsoup4>=4.12.0",
    "duckduckgo-search>=4.0.0",
    
    # 科学计算
    "numpy>=1.26.0",
    "scikit-learn>=1.4.0",
    
    # 工具
    "python-dotenv>=1.0.0",
    "tenacity>=8.2.0",
    
    # 开发
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
pythonpath = ["src"]
```

**`backend/requirements.txt`**（备选，如果用 pip）

```
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
python-socketio>=5.11.0
websockets>=12.0
pydantic>=2.6.0
pydantic-settings>=2.1.0
sqlalchemy[asyncio]>=2.0.0
asyncpg>=0.29.0
alembic>=1.13.0
redis>=5.0.0
openai>=1.12.0
httpx>=0.27.0
aiohttp>=3.9.0
beautifulsoup4>=4.12.0
duckduckgo-search>=4.0.0
numpy>=1.26.0
scikit-learn>=1.4.0
python-dotenv>=1.0.0
tenacity>=8.2.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
pytest-cov>=4.1.0
```

### 2.3 Docker Compose 配置

**`docker-compose.yml`**（项目根目录）

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:16-alpine
    container_name: travel-agent-db
    environment:
      POSTGRES_USER: travelagent
      POSTGRES_PASSWORD: travelagent123
      POSTGRES_DB: travel_agent
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U travelagent -d travel_agent"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: travel-agent-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: travel-agent-backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://travelagent:travelagent123@postgres:5432/travel_agent
      - REDIS_URL=redis://redis:6379/0
      - LLM_API_KEY=${LLM_API_KEY}
      - LLM_BASE_URL=${LLM_BASE_URL}
      - LLM_MODEL=${LLM_MODEL}
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./backend/src:/app/src
    command: uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    container_name: travel-agent-frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
      - NEXT_PUBLIC_WS_URL=ws://localhost:8000
    depends_on:
      - backend
    volumes:
      - ./frontend/src:/app/src
      - ./frontend/public:/app/public
    command: npm run dev

volumes:
  postgres_data:
  redis_data:
```

**`backend/Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY src/ ./src/

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**`frontend/Dockerfile`**

```dockerfile
FROM node:20-alpine

WORKDIR /app

# 复制依赖文件
COPY package*.json ./
RUN npm install

# 复制代码
COPY . .

# 暴露端口
EXPOSE 3000

# 启动命令
CMD ["npm", "run", "dev"]
```

---

## 三、数据库设计

### 3.1 ER 图

```
┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
│     users        │       │    sessions      │       │  conversations   │
├──────────────────┤       ├──────────────────┤       ├──────────────────┤
│ id (PK)          │──┐    │ id (PK)          │◄─────│ session_id (FK)  │
│ username         │  └───►│ user_id (FK)     │       │ user_message     │
│ created_at       │       │ destination      │       │ assistant_response│
└──────────────────┘       │ travel_days      │       │ intent           │
                           │ travel_dates     │       │ tools_used       │
                           │ status           │       │ timestamp        │
                           │ created_at       │       └──────────────────┘
                           │ updated_at       │
                           └──────────────────┘
                                    │
                                    │
                                    ▼
                           ┌──────────────────┐
                           │   itineraries    │
                           ├──────────────────┤
                           │ id (PK)          │
                           │ session_id (FK)  │
                           │ user_id (FK)     │
                           │ destination      │
                           │ travel_days      │
                           │ daily_plans (JSON)
                           │ preference_snapshot (JSON)
                           │ budget_snapshot (JSON)
                           │ status           │
                           │ created_at       │
                           └──────────────────┘
                                    │
                                    │
                                    ▼
                           ┌──────────────────┐
                           │ preference_changes│
                           ├──────────────────┤
                           │ id (PK)          │
                           │ session_id (FK)  │
                           │ field            │
                           │ old_value        │
                           │ new_value        │
                           │ timestamp        │
                           └──────────────────┘
```

### 3.2 PostgreSQL 建表语句

**`backend/src/config/schema.sql`**

```sql
-- 用户表
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) NOT NULL UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 会话表
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    destination VARCHAR(50),
    travel_days INTEGER,
    travel_dates VARCHAR(100),
    status VARCHAR(20) DEFAULT 'active', -- active, completed, abandoned
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 对话记录表
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    turn_number INTEGER NOT NULL,
    user_message TEXT NOT NULL,
    assistant_response TEXT,
    intent VARCHAR(50),
    intent_confidence FLOAT,
    extracted_entities JSONB DEFAULT '{}',
    tools_used JSONB DEFAULT '[]',
    agents_involved JSONB DEFAULT '[]',
    display_type VARCHAR(20) DEFAULT 'text', -- text, itinerary, budget_update, preference_update
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 行程表
CREATE TABLE IF NOT EXISTS itineraries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    destination VARCHAR(50) NOT NULL,
    travel_days INTEGER NOT NULL,
    travel_dates VARCHAR(100),
    daily_plans JSONB NOT NULL DEFAULT '[]',
    preference_snapshot JSONB DEFAULT '{}',
    budget_snapshot JSONB DEFAULT '{}',
    status VARCHAR(20) DEFAULT 'draft', -- draft, confirmed, completed, cancelled
    confirmed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 偏好变更记录表
CREATE TABLE IF NOT EXISTS preference_changes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    field VARCHAR(50) NOT NULL,
    old_value JSONB,
    new_value JSONB,
    source_message TEXT,
    turn_id UUID,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp);
CREATE INDEX IF NOT EXISTS idx_itineraries_session_id ON itineraries(session_id);
CREATE INDEX IF NOT EXISTS idx_itineraries_user_id ON itineraries(user_id);
CREATE INDEX IF NOT EXISTS idx_preference_changes_session_id ON preference_changes(session_id);
```

### 3.3 SQLAlchemy 模型定义

**`backend/src/core/models.py`**

```python
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import (
    String, Integer, Float, Text, DateTime, ForeignKey, JSON
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    
    sessions: Mapped[List["Session"]] = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    itineraries: Mapped[List["Itinerary"]] = relationship("Itinerary", back_populates="user", cascade="all, delete-orphan")


class Session(Base):
    __tablename__ = "sessions"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    destination: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    travel_days: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    travel_dates: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user: Mapped[Optional["User"]] = relationship("User", back_populates="sessions")
    conversations: Mapped[List["Conversation"]] = relationship("Conversation", back_populates="session", cascade="all, delete-orphan")
    itineraries: Mapped[List["Itinerary"]] = relationship("Itinerary", back_populates="session", cascade="all, delete-orphan")
    preference_changes: Mapped[List["PreferenceChange"]] = relationship("PreferenceChange", back_populates="session", cascade="all, delete-orphan")


class Conversation(Base):
    __tablename__ = "conversations"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("sessions.id"))
    turn_number: Mapped[int] = mapped_column(Integer, nullable=False)
    user_message: Mapped[str] = mapped_column(Text, nullable=False)
    assistant_response: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    intent: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    intent_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    extracted_entities: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, default=dict)
    tools_used: Mapped[Optional[List[str]]] = mapped_column(JSON, default=list)
    agents_involved: Mapped[Optional[List[str]]] = mapped_column(JSON, default=list)
    display_type: Mapped[str] = mapped_column(String(20), default="text")
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    
    session: Mapped["Session"] = relationship("Session", back_populates="conversations")


class Itinerary(Base):
    __tablename__ = "itineraries"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("sessions.id"))
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    destination: Mapped[str] = mapped_column(String(50), nullable=False)
    travel_days: Mapped[int] = mapped_column(Integer, nullable=False)
    travel_dates: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    daily_plans: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, default=list)
    preference_snapshot: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, default=dict)
    budget_snapshot: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(20), default="draft")
    confirmed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    
    session: Mapped["Session"] = relationship("Session", back_populates="itineraries")
    user: Mapped[Optional["User"]] = relationship("User", back_populates="itineraries")


class PreferenceChange(Base):
    __tablename__ = "preference_changes"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("sessions.id"))
    field: Mapped[str] = mapped_column(String(50), nullable=False)
    old_value: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    new_value: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    source_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    turn_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    
    session: Mapped["Session"] = relationship("Session", back_populates="preference_changes")
```

### 3.4 Pydantic 数据模型

**`backend/src/core/schemas.py`**

```python
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field


# ============ 基础模型 ============

class Location(BaseModel):
    lat: float
    lng: float


class POI(BaseModel):
    id: Optional[str] = None
    name: str
    city: str
    category: Literal["attraction", "restaurant", "shopping", "hotel", "entertainment", "other"]
    subcategory: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    location: Optional[Location] = None
    rating: Optional[float] = None
    review_count: Optional[int] = None
    recommended_duration_min: Optional[int] = None
    ticket_price: Optional[float] = None
    average_meal_cost: Optional[float] = None
    opening_hours: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    source: Optional[str] = None


class ScoredPOI(POI):
    base_score: float = 0.0
    preference_score: float = 0.0
    final_score: float = 0.0


class WeatherDay(BaseModel):
    date: str
    condition: str
    temperature_high: int
    temperature_low: int
    precipitation_probability: Optional[int] = None
    wind_speed: Optional[int] = None
    suggestion: Optional[str] = None


# ============ 用户画像模型 ============

class UserProfile(BaseModel):
    destination: Optional[str] = None
    travel_days: Optional[int] = None
    travel_dates: Optional[str] = None
    travelers_count: Optional[int] = Field(default=1)
    travelers_type: Optional[Literal["alone", "couple", "family", "friends", "parents", "other"]] = None
    budget_range: Optional[float] = None
    food_preferences: List[str] = Field(default_factory=list)
    interests: List[str] = Field(default_factory=list)
    pace: Optional[Literal["relaxed", "moderate", "intensive"]] = Field(default="moderate")
    accommodation_preference: Optional[str] = None
    special_requests: List[str] = Field(default_factory=list)
    preference_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    class Config:
        populate_by_name = True


# ============ 行程模型 ============

class Activity(BaseModel):
    poi_name: str
    poi_id: Optional[str] = None
    category: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_min: int
    ticket_price: Optional[float] = None
    meal_cost: Optional[float] = None
    transport_cost: Optional[float] = None
    location: Optional[Location] = None
    recommendation_reason: Optional[str] = None
    transit_from_prev: Optional[Dict[str, Any]] = None


class DayPlan(BaseModel):
    day_number: int
    date: Optional[str] = None
    theme: Optional[str] = None
    activities: List[Activity] = Field(default_factory=list)
    total_cost: float = 0.0
    total_walking_steps: int = 0
    total_transit_time_min: int = 0


# ============ 面板模型 ============

class BudgetPanel(BaseModel):
    total_budget: Optional[float] = None
    spent: float = 0.0
    remaining: Optional[float] = None
    breakdown: Dict[str, float] = Field(default_factory=lambda: {
        "accommodation": 0, "meals": 0, "transport": 0,
        "tickets": 0, "shopping": 0, "buffer": 0
    })
    status: Literal["within_budget", "over_budget", "unknown"] = "unknown"


class PreferencePanel(BaseModel):
    destination: Optional[str] = None
    travel_days: Optional[int] = None
    travel_dates: Optional[str] = None
    travelers_count: Optional[int] = None
    travelers_type: Optional[str] = None
    budget_range: Optional[float] = None
    food_preferences: List[str] = Field(default_factory=list)
    interests: List[str] = Field(default_factory=list)
    pace: Optional[str] = None
    special_requests: List[str] = Field(default_factory=list)


# ============ 意图识别模型 ============

class PreferenceChange(BaseModel):
    field: str
    old_value: Any
    new_value: Any
    change_type: Literal["update", "add", "remove"] = "update"


class IntentResult(BaseModel):
    intent: Literal[
        "generate_itinerary", "modify_itinerary", "update_preferences",
        "query_info", "confirm_itinerary", "view_history", "chitchat", "unknown"
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    user_entities: Dict[str, Any] = Field(default_factory=dict)
    missing_required: List[str] = Field(default_factory=list)
    missing_recommended: List[str] = Field(default_factory=list)
    preference_changes: List[PreferenceChange] = Field(default_factory=list)
    is_fuzzy: bool = False
    clarification_questions: List[str] = Field(default_factory=list)
    reasoning: Optional[str] = None


# ============ 对话模型 ============

class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: Optional[datetime] = None


class ConversationTurn(BaseModel):
    turn_id: str
    session_id: str
    user_message: str
    assistant_response: Optional[str] = None
    intent: Optional[str] = None
    intent_confidence: Optional[float] = None
    tools_used: List[str] = Field(default_factory=list)
    agents_involved: List[str] = Field(default_factory=list)
    timestamp: str
    display_type: Literal["text", "itinerary", "budget_update", "preference_update"] = "text"


# ============ 行程记录模型 ============

class ItineraryRecord(BaseModel):
    record_id: str
    session_id: str
    destination: str
    travel_days: int
    travel_dates: Optional[str] = None
    daily_plans: List[Dict[str, Any]] = Field(default_factory=list)
    preference_snapshot: Optional[Dict[str, Any]] = None
    budget_snapshot: Optional[Dict[str, Any]] = None
    status: Literal["draft", "confirmed", "completed", "cancelled"] = "draft"
    confirmed_at: Optional[str] = None
    created_at: str


# ============ Agent 通信模型 ============

class AgentMessage(BaseModel):
    message_id: str
    correlation_id: str
    sender: str
    receiver: str
    message_type: Literal["request", "response", "event"]
    action: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str
    deadline_ms: int = 30000


class AgentTaskResult(BaseModel):
    task_id: str
    status: Literal["completed", "failed", "partial"]
    result: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    execution_time_ms: int = 0
    trace: List[str] = Field(default_factory=list)
    fallback_used: bool = False


# ============ WebSocket 消息模型 ============

class WSMessage(BaseModel):
    type: Literal[
        "user_message", "assistant_stream", "assistant_done",
        "itinerary_update", "budget_update", "preference_update",
        "intent_recognized", "error"
    ]
    content: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


# ============ API 请求/响应模型 ============

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    response: str
    intent: Optional[str] = None
    itinerary: Optional[List[DayPlan]] = None
    budget_panel: Optional[BudgetPanel] = None
    preference_panel: Optional[PreferencePanel] = None


class ConfirmItineraryRequest(BaseModel):
    session_id: str
    itinerary_id: Optional[str] = None


class ItineraryResponse(BaseModel):
    itinerary: List[DayPlan]
    budget: BudgetPanel
    preferences: PreferencePanel


# ============ 爬虫模型 ============

class SearchResult(BaseModel):
    title: str
    url: str
    snippet: Optional[str] = None
    source: Optional[str] = None


class CrawledPage(BaseModel):
    url: str
    content: str
    status: int
    title: Optional[str] = None


class ExtractedPOI(BaseModel):
    name: str
    category: Optional[str] = None
    description: Optional[str] = None
    highlights: Optional[str] = None
    rating: Optional[float] = None
    price_range: Optional[str] = None
    address: Optional[str] = None
    source_url: Optional[str] = None


# ============ 校验模型 ============

class ValidationCheck(BaseModel):
    check_name: str
    passed: bool
    score: float
    details: Optional[Dict[str, Any]] = None


class ValidationResult(BaseModel):
    passed: bool
    checks: Dict[str, ValidationCheck] = Field(default_factory=dict)
    overall_score: float = 0.0
    improvement_suggestions: List[str] = Field(default_factory=list)


# ============ 配置模型 ============

class Settings(BaseModel):
    database_url: str = "postgresql+asyncpg://travelagent:travelagent123@localhost:5432/travel_agent"
    redis_url: str = "redis://localhost:6379/0"
    llm_api_key: str
    llm_base_url: str = "https://api.anthropic.com/v1"
    llm_model: str = "claude-sonnet-4-6"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 4096
    
    # 爬虫配置
    crawler_rate_limit: float = 1.0
    crawler_max_retries: int = 3
    crawler_timeout: int = 10
    
    # 缓存配置
    cache_ttl_seconds: int = 3600
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
```

---

## 四、后端技术方案

### 4.1 配置文件

**`backend/src/config/settings.py`**

```python
import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # 数据库
    database_url: str = "postgresql+asyncpg://travelagent:travelagent123@localhost:5432/travel_agent"
    redis_url: str = "redis://localhost:6379/0"
    
    # LLM
    llm_api_key: str = ""
    llm_base_url: str = "https://api.anthropic.com/v1"
    llm_model: str = "claude-sonnet-4-6"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 4096
    
    # 爬虫
    crawler_rate_limit: float = 1.0
    crawler_max_retries: int = 3
    crawler_timeout: int = 10
    
    # 缓存
    cache_ttl_seconds: int = 3600
    
    # 应用
    app_env: str = "development"
    debug: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
```

**`backend/.env`**（开发环境配置模板）

```env
DATABASE_URL=postgresql+asyncpg://travelagent:travelagent123@localhost:5432/travel_agent
REDIS_URL=redis://localhost:6379/0
LLM_API_KEY=your-api-key-here
LLM_BASE_URL=https://api.anthropic.com/v1
LLM_MODEL=claude-sonnet-4-6
APP_ENV=development
DEBUG=true
```

### 4.2 数据库连接管理

**`backend/src/core/database.py`**

```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from src.config.settings import get_settings

settings = get_settings()

# 创建异步引擎
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    future=True,
    pool_size=10,
    max_overflow=20
)

# 创建异步会话工厂
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)


async def get_db() -> AsyncSession:
    """FastAPI 依赖注入用数据库会话"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """初始化数据库表（仅开发环境使用）"""
    from src.core.models import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
```

### 4.3 Redis 连接管理

**`backend/src/core/redis_client.py`**

```python
import json
from typing import Optional, Any
import redis.asyncio as redis
from src.config.settings import get_settings

settings = get_settings()

_redis_pool: Optional[redis.Redis] = None


async def get_redis() -> redis.Redis:
    """获取 Redis 连接（单例）"""
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True
        )
    return _redis_pool


async def close_redis():
    """关闭 Redis 连接"""
    global _redis_pool
    if _redis_pool:
        await _redis_pool.close()
        _redis_pool = None


class StateStore:
    """基于 Redis 的 State 存储"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.ttl = 7 * 24 * 3600  # 7 天
    
    async def get(self, key: str) -> Optional[Any]:
        data = await self.redis.get(key)
        return json.loads(data) if data else None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        await self.redis.setex(key, ttl or self.ttl, json.dumps(value, default=str))
    
    async def delete(self, key: str):
        await self.redis.delete(key)
```

### 4.4 LLM 客户端封装

**`backend/src/core/llm_client.py`**

```python
import json
from typing import Optional, Type, Any
from openai import AsyncOpenAI
from pydantic import BaseModel
from src.config.settings import get_settings

settings = get_settings()


class LLMClient:
    """
    LLM 客户端封装。
    支持结构化输出（通过 JSON mode）。
    """
    
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url
        )
        self.model = settings.llm_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
    
    async def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """普通对话调用"""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens
        )
        return response.choices[0].message.content
    
    async def structured_call(
        self,
        messages: list[dict],
        schema: Type[BaseModel],
        temperature: Optional[float] = None
    ) -> dict:
        """
        结构化输出调用。
        要求 LLM 返回符合指定 Pydantic Schema 的 JSON。
        """
        # 在 system message 中注入 JSON 格式要求
        schema_description = self._generate_schema_description(schema)
        
        system_msg = {
            "role": "system",
            "content": f"""You must respond with a valid JSON object that matches the following schema:
{schema_description}

Important: Respond ONLY with the JSON object. Do not include markdown formatting, explanations, or any other text."""
        }
        
        # 确保 system message 在最前面
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] += "\n\n" + system_msg["content"]
        else:
            messages = [system_msg] + messages
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        
        # 解析 JSON
        try:
            data = json.loads(content)
            # 验证 schema
            validated = schema(**data)
            return validated.model_dump()
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON: {content[:200]}") from e
        except Exception as e:
            raise ValueError(f"LLM output does not match schema: {e}") from e
    
    def _generate_schema_description(self, schema: Type[BaseModel]) -> str:
        """生成 schema 的自然语言描述"""
        return json.dumps(schema.model_json_schema(), indent=2, ensure_ascii=False)


# 全局单例
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
```

### 4.5 EventBus（事件总线）

**`backend/src/core/event_bus.py`**

```python
import asyncio
from typing import Dict, List, Callable, Any
from collections import defaultdict


class EventBus:
    """
    轻量级异步事件总线。
    Agent 间通过发布-订阅模式通信，解耦调用关系。
    """
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
    
    def subscribe(self, event_type: str, handler: Callable):
        """订阅事件"""
        self._subscribers[event_type].append(handler)
    
    def unsubscribe(self, event_type: str, handler: Callable):
        """取消订阅"""
        if handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)
    
    async def publish(self, event_type: str, payload: Dict[str, Any]):
        """发布事件，异步通知所有订阅者"""
        handlers = self._subscribers.get(event_type, [])
        if not handlers:
            return
        
        # 并行执行所有订阅者
        tasks = []
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    tasks.append(asyncio.create_task(handler(payload)))
                else:
                    handler(payload)
            except Exception as e:
                print(f"Event handler error for {event_type}: {e}")
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def clear(self):
        """清除所有订阅"""
        self._subscribers.clear()


# 全局事件总线实例
event_bus = EventBus()
```

### 4.6 State Manager

**`backend/src/core/state_manager.py`**

```python
import json
import uuid
from typing import Optional
from datetime import datetime
from src.core.schemas import TravelState, UserProfile, Message
from src.core.redis_client import get_redis, StateStore


class StateManager:
    """
    状态管理器：管理单会话内的状态。
    使用 Redis 做快速读写。
    """
    
    def __init__(self):
        self.store: Optional[StateStore] = None
    
    async def _get_store(self) -> StateStore:
        if self.store is None:
            redis = await get_redis()
            self.store = StateStore(redis)
        return self.store
    
    async def create_session(self, user_id: Optional[str] = None) -> str:
        """创建新会话"""
        session_id = str(uuid.uuid4())
        state = TravelState(
            session_id=session_id,
            user_id=user_id,
            user_profile=UserProfile(),
            recent_messages=[],
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat()
        )
        await self.save(state)
        return session_id
    
    async def load(self, session_id: str) -> TravelState:
        """加载 State"""
        store = await self._get_store()
        data = await store.get(f"state:{session_id}")
        
        if data:
            return TravelState(**data)
        
        # 新会话
        return TravelState(
            session_id=session_id,
            user_profile=UserProfile(),
            recent_messages=[],
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat()
        )
    
    async def save(self, state: TravelState):
        """保存 State"""
        store = await self._get_store()
        state.updated_at = datetime.utcnow().isoformat()
        state.turn_count += 1
        await store.set(f"state:{state.session_id}", state.model_dump())
    
    async def add_message(self, session_id: str, role: str, content: str):
        """添加对话消息"""
        state = await self.load(session_id)
        state.recent_messages.append(Message(role=role, content=content))
        
        # 只保留最近 20 轮
        if len(state.recent_messages) > 40:
            state.recent_messages = state.recent_messages[-40:]
        
        await self.save(state)
    
    async def update_profile(self, session_id: str, **kwargs):
        """更新用户画像"""
        state = await self.load(session_id)
        for key, value in kwargs.items():
            if hasattr(state.user_profile, key):
                setattr(state.user_profile, key, value)
        await self.save(state)
```

### 4.7 Skill 层实现

#### 4.7.1 Web Search Skill

**`backend/src/skills/web_search.py`**

```python
from typing import List
from duckduckgo_search import DDGS
from src.core.schemas import SearchResult


class WebSearchSkill:
    """
    网络搜索 Skill。
    使用 DuckDuckGo 搜索引擎（无需 API Key）。
    """
    
    async def search(
        self,
        query: str,
        top_n: int = 10,
        region: str = "cn-zh"
    ) -> List[SearchResult]:
        """
        执行网络搜索。
        
        Args:
            query: 搜索关键词
            top_n: 返回结果数量
            region: 搜索区域
        
        Returns:
            搜索结果列表
        """
        try:
            with DDGS() as ddgs:
                results = ddgs.text(
                    query,
                    region=region,
                    max_results=top_n
                )
                
                return [
                    SearchResult(
                        title=r.get("title", ""),
                        url=r.get("href", ""),
                        snippet=r.get("body", ""),
                        source="duckduckgo"
                    )
                    for r in results
                ]
        except Exception as e:
            print(f"Web search error: {e}")
            return []
```

#### 4.7.2 Web Crawler Skill

**`backend/src/skills/web_crawler.py`**

```python
import asyncio
import random
from typing import Optional
import aiohttp
from bs4 import BeautifulSoup
from src.core.schemas import CrawledPage
from src.config.settings import get_settings

settings = get_settings()

# User-Agent 池
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
]


class WebCrawlerSkill:
    """
    网页爬取 Skill。
    异步爬取网页内容，提取正文。
    """
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limit = settings.crawler_rate_limit
        self.last_request_time = 0
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                headers={"User-Agent": random.choice(USER_AGENTS)},
                timeout=aiohttp.ClientTimeout(total=settings.crawler_timeout)
            )
        return self.session
    
    async def crawl(self, url: str) -> CrawledPage:
        """
        爬取单个页面。
        
        Args:
            url: 目标 URL
        
        Returns:
            爬取结果
        """
        # 速率限制
        await self._rate_limit()
        
        session = await self._get_session()
        
        try:
            async with session.get(url) as response:
                html = await response.text()
                
                # 解析正文
                content = self._extract_content(html)
                title = self._extract_title(html)
                
                return CrawledPage(
                    url=url,
                    content=content,
                    status=response.status,
                    title=title
                )
        except asyncio.TimeoutError:
            return CrawledPage(url=url, content="", status=408, title="")
        except Exception as e:
            return CrawledPage(url=url, content=f"Error: {str(e)}", status=500, title="")
    
    async def crawl_many(self, urls: List[str]) -> List[CrawledPage]:
        """并行爬取多个页面（带并发限制）"""
        semaphore = asyncio.Semaphore(3)  # 最多 3 个并发
        
        async def crawl_with_limit(url: str) -> CrawledPage:
            async with semaphore:
                return await self.crawl(url)
        
        tasks = [crawl_with_limit(url) for url in urls]
        return await asyncio.gather(*tasks)
    
    def _extract_content(self, html: str) -> str:
        """提取页面正文"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # 移除脚本和样式
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # 尝试获取主要内容
        for selector in ["main", "article", "[role='main']", ".content", "#content"]:
            element = soup.select_one(selector)
            if element:
                return element.get_text(separator='\n', strip=True)[:10000]
        
        # 回退到 body
        body = soup.find('body')
        if body:
            return body.get_text(separator='\n', strip=True)[:10000]
        
        return soup.get_text(separator='\n', strip=True)[:10000]
    
    def _extract_title(self, html: str) -> Optional[str]:
        """提取页面标题"""
        soup = BeautifulSoup(html, 'html.parser')
        title = soup.find('title')
        return title.get_text(strip=True) if title else None
    
    async def _rate_limit(self):
        """速率限制"""
        import time
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.rate_limit:
            await asyncio.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()
    
    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
```

#### 4.7.3 POI 搜索 Skill

**`backend/src/skills/poi_search.py`**

```python
from typing import List, Optional
from src.core.llm_client import get_llm_client
from src.core.schemas import ScoredPOI, ExtractedPOI
from src.skills.web_search import WebSearchSkill
from src.skills.web_crawler import WebCrawlerSkill


class POISearchSkill:
    """
    POI 搜索 Skill。
    通过网络搜索 + 爬虫 + LLM 提取，获取城市 POI 信息。
    """
    
    def __init__(self):
        self.search_skill = WebSearchSkill()
        self.crawler_skill = WebCrawlerSkill()
        self.llm = get_llm_client()
    
    async def search_pois(
        self,
        city: str,
        keywords: List[str],
        category: Optional[str] = None,
        top_n: int = 20
    ) -> List[ScoredPOI]:
        """
        搜索城市 POI。
        
        策略：
        1. 并行搜索多个关键词
        2. 爬取搜索结果页面
        3. LLM 提取结构化 POI 数据
        4. 去重 + 评分
        """
        # 构建搜索查询
        queries = [
            f"{city} 必去景点 推荐 2026",
            f"{city} 美食推荐",
            f"{city} 网红打卡 拍照",
        ]
        
        # 添加关键词特定查询
        for kw in keywords[:3]:
            queries.append(f"{city} {kw} 推荐")
        
        # 并行搜索
        search_tasks = [self.search_skill.search(q, top_n=5) for q in queries[:5]]
        search_results = await asyncio.gather(*search_tasks)
        
        # 收集所有 URL
        all_urls = []
        for results in search_results:
            for r in results:
                if r.url and r.url.startswith("http"):
                    all_urls.append(r.url)
        
        # 去重
        all_urls = list(dict.fromkeys(all_urls))[:15]
        
        # 并行爬取
        pages = await self.crawler_skill.crawl_many(all_urls)
        
        # LLM 提取 POI
        poi_tasks = [self._extract_pois_from_page(page, city) for page in pages if page.status == 200]
        extracted_lists = await asyncio.gather(*poi_tasks)
        
        # 合并去重
        all_pois = []
        seen_names = set()
        for pois in extracted_lists:
            for poi in pois:
                if poi.name not in seen_names:
                    seen_names.add(poi.name)
                    all_pois.append(ScoredPOI(
                        name=poi.name,
                        city=city,
                        category=poi.category or "other",
                        description=poi.description,
                        rating=poi.rating,
                        tags=[],
                        base_score=0.5
                    ))
        
        return all_pois[:top_n]
    
    async def _extract_pois_from_page(self, page, city: str) -> List[ExtractedPOI]:
        """使用 LLM 从网页提取 POI 列表"""
        prompt = f"""从以下关于 {city} 的网页内容中，提取所有提到的景点、餐厅、购物场所等 POI。

网页内容：
{page.content[:5000]}

请提取每个 POI 的以下信息：
- name: 名称（必需）
- category: 类别（attraction/restaurant/shopping/hotel/entertainment/other）
- description: 简短描述
- rating: 评分（如果有）
- highlights: 特色/亮点

以 JSON 数组格式输出：
[{{"name": "...", "category": "...", ...}}, ...]
"""
        
        try:
            response = await self.llm.chat([
                {"role": "user", "content": prompt}
            ], temperature=0.3)
            
            import json
            data = json.loads(response)
            if isinstance(data, list):
                return [ExtractedPOI(**item) for item in data]
            elif isinstance(data, dict) and "pois" in data:
                return [ExtractedPOI(**item) for item in data["pois"]]
            return []
        except Exception as e:
            print(f"POI extraction error: {e}")
            return []
```

### 4.8 Agent 层实现

#### 4.8.1 Intent Recognition Agent

**`backend/src/agents/intent_recognition.py`**

```python
from typing import List
from src.core.llm_client import LLMClient
from src.core.schemas import IntentResult, PreferenceChange, TravelState, Message


INTENT_RECOGNITION_PROMPT = """你是一个旅行规划系统的意图识别专家。分析用户的输入，判断意图并提取关键信息。

## 可识别的意图类型
1. generate_itinerary - 生成新行程
2. modify_itinerary - 修改已有行程
3. update_preferences - 更新偏好
4. query_info - 查询信息
5. confirm_itinerary - 确认行程
6. view_history - 查看历史
7. chitchat - 闲聊

## 关键信息
必需：destination（目的地）, travel_days（天数）, travel_dates（日期）
重要：travelers_count（人数）, budget_range（预算）, travelers_type（同行类型）
偏好：food_preferences（饮食）, interests（兴趣）, pace（节奏）, special_requests（特殊要求）

## 判断规则
1. 缺少 destination/travel_days/travel_dates 则标记 missing_required
2. 检测用户是否更新了偏好（与当前偏好对比）
3. 置信度<0.7时标记需要确认

以JSON格式输出，字段：intent, confidence, user_entities, missing_required, missing_recommended, preference_changes, clarification_questions, reasoning"""


class IntentRecognitionAgent:
    """意图识别 Agent：LLM 驱动"""
    
    def __init__(self, llm: LLMClient):
        self.llm = llm
    
    async def recognize(
        self,
        user_input: str,
        conversation_history: List[Message],
        current_state: TravelState
    ) -> IntentResult:
        # 构建 messages
        messages = [
            {"role": "system", "content": INTENT_RECOGNITION_PROMPT},
            {"role": "system", "content": f"当前用户画像：{current_state.user_profile.model_dump_json(ensure_ascii=False)}"},
        ]
        
        # 添加最近对话历史
        for msg in conversation_history[-10:]:
            messages.append({"role": msg.role, "content": msg.content})
        
        messages.append({"role": "user", "content": user_input})
        
        # 调用 LLM（结构化输出）
        result = await self.llm.structured_call(
            messages=messages,
            schema=IntentResult,
            temperature=0.3
        )
        
        intent_result = IntentResult(**result)
        
        # 检测偏好变更
        intent_result.preference_changes = self._detect_preference_changes(
            intent_result.user_entities, current_state.user_profile
        )
        
        return intent_result
    
    def _detect_preference_changes(self, new_entities: dict, current_profile) -> List[PreferenceChange]:
        changes = []
        
        # food_preferences
        new_food = set(new_entities.get("food_preferences", []) or [])
        old_food = set(current_profile.food_preferences or [])
        if new_food != old_food:
            changes.append(PreferenceChange(
                field="food_preferences",
                old_value=list(old_food),
                new_value=list(new_food)
            ))
        
        # interests
        new_interests = set(new_entities.get("interests", []) or [])
        old_interests = set(current_profile.interests or [])
        if new_interests != old_interests:
            changes.append(PreferenceChange(
                field="interests",
                old_value=list(old_interests),
                new_value=list(new_interests)
            ))
        
        # budget
        new_budget = new_entities.get("budget_range")
        old_budget = current_profile.budget_range
        if new_budget is not None and new_budget != old_budget:
            changes.append(PreferenceChange(
                field="budget_range",
                old_value=old_budget,
                new_value=new_budget
            ))
        
        # pace
        new_pace = new_entities.get("pace")
        old_pace = current_profile.pace
        if new_pace is not None and new_pace != old_pace:
            changes.append(PreferenceChange(
                field="pace",
                old_value=old_pace,
                new_value=new_pace
            ))
        
        return changes
```

#### 4.8.2 Preference & Budget Agent

**`backend/src/agents/preference_budget.py`**

```python
from typing import List
from src.core.schemas import (
    UserProfile, BudgetPanel, PreferencePanel, 
    PreferenceChange, DayPlan
)
from src.core.state_manager import StateManager
from src.core.event_bus import event_bus


class PreferenceBudgetAgent:
    """偏好与预算管理 Agent"""
    
    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
    
    async def update_preferences(
        self,
        session_id: str,
        changes: List[PreferenceChange]
    ) -> UserProfile:
        """更新用户偏好"""
        state = await self.state_manager.load(session_id)
        profile = state.user_profile
        
        for change in changes:
            if hasattr(profile, change.field):
                setattr(profile, change.field, change.new_value)
            
            # 记录变更历史
            profile.preference_history.append({
                "timestamp": __import__('datetime').datetime.utcnow().isoformat(),
                "field": change.field,
                "old": change.old_value,
                "new": change.new_value
            })
        
        await self.state_manager.save(state)
        
        # 发布偏好变更事件
        await event_bus.publish("PreferenceChanged", {
            "session_id": session_id,
            "changes": [c.model_dump() for c in changes]
        })
        
        return profile
    
    def calculate_budget_breakdown(
        self,
        itinerary: List[DayPlan],
        profile: UserProfile
    ) -> BudgetPanel:
        """计算预算分项"""
        breakdown = {
            "accommodation": 0,
            "meals": 0,
            "transport": 0,
            "tickets": 0,
            "shopping": 0,
            "buffer": 0
        }
        
        for day in itinerary:
            breakdown["tickets"] += sum(
                (a.ticket_price or 0) for a in day.activities
            )
            breakdown["meals"] += sum(
                (a.meal_cost or 0) for a in day.activities
            )
            breakdown["transport"] += sum(
                (a.transport_cost or 0) for a in day.activities
            )
        
        # 住宿估算（简化版）
        if profile.travel_days:
            city_factor = self._get_city_factor(profile.destination)
            breakdown["accommodation"] = profile.travel_days * 300 * city_factor
        
        # buffer（10-15%）
        subtotal = sum(breakdown.values())
        breakdown["buffer"] = subtotal * 0.12
        
        total = sum(breakdown.values())
        budget = profile.budget_range
        
        return BudgetPanel(
            total_budget=budget,
            spent=total,
            remaining=budget - total if budget else None,
            breakdown=breakdown,
            status="within_budget" if (not budget or total <= budget) else "over_budget"
        )
    
    def get_panel_data(self, profile: UserProfile) -> PreferencePanel:
        """生成偏好面板数据"""
        return PreferencePanel(
            destination=profile.destination,
            travel_days=profile.travel_days,
            travel_dates=profile.travel_dates,
            travelers_count=profile.travelers_count,
            travelers_type=profile.travelers_type,
            budget_range=profile.budget_range,
            food_preferences=profile.food_preferences or [],
            interests=profile.interests or [],
            pace=profile.pace,
            special_requests=profile.special_requests or []
        )
    
    def _get_city_factor(self, city: str) -> float:
        """城市消费系数"""
        tier1 = {"北京", "上海", "广州", "深圳", "杭州"}
        tier2 = {"成都", "重庆", "西安", "苏州", "南京", "武汉", "长沙", "厦门", "青岛"}
        
        if city in tier1:
            return 1.3
        elif city in tier2:
            return 1.0
        else:
            return 0.8
```

#### 4.8.3 Itinerary Planner Agent

**`backend/src/agents/itinerary_planner.py`**

```python
import random
from typing import List, Optional, Tuple
import numpy as np
from sklearn.cluster import DBSCAN

from src.core.schemas import (
    ScoredPOI, DayPlan, Activity, UserProfile, WeatherDay, Location
)


class ItineraryPlannerAgent:
    """行程规划 Agent"""
    
    def __init__(self):
        pass
    
    def plan(
        self,
        pois: List[ScoredPOI],
        profile: UserProfile,
        weather: Optional[List[WeatherDay]] = None
    ) -> List[DayPlan]:
        """生成行程"""
        
        # Step 1: POI 打分
        scored = self._score_pois(pois, profile)
        
        # Step 2: 区域聚类
        clusters, noise = self._cluster_pois(scored)
        
        # Step 3: 日分配
        num_days = profile.travel_days or 3
        days = self._assign_to_days(clusters, noise, num_days, profile)
        
        # Step 4: 日内排序
        for day in days:
            day.activities = self._sort_activities(day.activities)
        
        # Step 5: 时间表生成
        for i, day in enumerate(days):
            day.day_number = i + 1
            day.date = self._get_date(profile.travel_dates, i)
            day.activities = self._build_schedule(day.activities, profile)
            day.total_cost = sum(
                (a.ticket_price or 0) + (a.meal_cost or 0) + (a.transport_cost or 0)
                for a in day.activities
            )
        
        return days
    
    def _score_pois(self, pois: List[ScoredPOI], profile: UserProfile) -> List[ScoredPOI]:
        """根据偏好给 POI 打分"""
        for poi in pois:
            score = poi.base_score
            
            # 兴趣匹配
            for interest in profile.interests or []:
                if interest in (poi.tags or []) or interest in (poi.description or ""):
                    score += 0.2
            
            # 饮食偏好
            if poi.category == "restaurant" and profile.food_preferences:
                for food in profile.food_preferences:
                    if food in (poi.tags or []) or food in (poi.description or ""):
                        score += 0.3
            
            # 节奏匹配
            if profile.pace == "relaxed":
                score += 0.05  # 轻微加分，不过滤
            
            poi.preference_score = min(1.0, score)
            poi.final_score = poi.preference_score
        
        return sorted(pois, key=lambda p: p.final_score, reverse=True)
    
    def _cluster_pois(self, pois: List[ScoredPOI]) -> Tuple[List[List[ScoredPOI]], List[ScoredPOI]]:
        """DBSCAN 区域聚类"""
        # 简化版：如果没有坐标，按类别聚类
        has_coords = all(p.location for p in pois if p.location)
        
        if has_coords:
            coords = np.array([[p.location.lat, p.location.lng] for p in pois])
            clustering = DBSCAN(eps=0.01, min_samples=2).fit(coords)
            
            clusters = {}
            noise = []
            for poi, label in zip(pois, clustering.labels_):
                if label == -1:
                    noise.append(poi)
                else:
                    clusters.setdefault(label, []).append(poi)
            
            return list(clusters.values()), noise
        else:
            # 按类别聚类
            by_category = {}
            for poi in pois:
                by_category.setdefault(poi.category, []).append(poi)
            return list(by_category.values()), []
    
    def _assign_to_days(
        self,
        clusters: List[List[ScoredPOI]],
        noise: List[ScoredPOI],
        num_days: int,
        profile: UserProfile
    ) -> List[DayPlan]:
        """将 POI 分配到各天"""
        days = [DayPlan(day_number=i+1, activities=[]) for i in range(num_days)]
        
        # 计算每天目标景点数（根据节奏）
        pace_map = {"relaxed": 3, "moderate": 4, "intensive": 5}
        target_per_day = pace_map.get(profile.pace, 4)
        
        # 合并所有 POI
        all_pois = []
        for cluster in sorted(clusters, key=len, reverse=True):
            all_pois.extend(cluster)
        all_pois.extend(noise)
        
        # 分配（轮询）
        for i, poi in enumerate(all_pois[:num_days * target_per_day]):
            day_idx = i % num_days
            days[day_idx].activities.append(Activity(
                poi_name=poi.name,
                category=poi.category,
                duration_min=poi.recommended_duration_min or 120,
                ticket_price=poi.ticket_price,
                recommendation_reason=poi.description
            ))
        
        return days
    
    def _sort_activities(self, activities: List[Activity]) -> List[Activity]:
        """日内排序（简化版：按类别排序）"""
        category_order = {"attraction": 0, "restaurant": 1, "shopping": 2, "entertainment": 3, "other": 4}
        return sorted(activities, key=lambda a: category_order.get(a.category, 99))
    
    def _build_schedule(self, activities: List[Activity], profile: UserProfile) -> List[Activity]:
        """生成时间表"""
        start_time = 9 * 60  # 9:00 开始（分钟数）
        
        for i, activity in enumerate(activities):
            # 计算开始时间
            if i == 0:
                activity.start_time = self._format_time(start_time)
            else:
                # 加上交通时间（简化：30分钟）
                start_time += 30
                activity.start_time = self._format_time(start_time)
            
            # 计算结束时间
            end_time = start_time + activity.duration_min
            activity.end_time = self._format_time(end_time)
            
            # 为下一段准备
            start_time = end_time
            
            # 午餐插入（简化：12:00-13:00）
            if start_time > 12 * 60 and start_time < 13 * 60:
                start_time = 13 * 60
        
        return activities
    
    def _format_time(self, minutes: int) -> str:
        """分钟数转 HH:MM"""
        h = minutes // 60
        m = minutes % 60
        return f"{h:02d}:{m:02d}"
    
    def _get_date(self, travel_dates: Optional[str], day_offset: int) -> Optional[str]:
        """获取具体日期（简化版）"""
        if not travel_dates:
            return None
        # TODO: 解析日期字符串
        return travel_dates
```

### 4.9 Orchestrator（编排器）

**`backend/src/core/orchestrator.py`**

```python
import asyncio
from typing import Optional
from src.core.llm_client import get_llm_client
from src.core.schemas import (
    TravelState, IntentResult, ChatResponse, 
    BudgetPanel, PreferencePanel, DayPlan
)
from src.core.state_manager import StateManager
from src.core.event_bus import event_bus
from src.agents.intent_recognition import IntentRecognitionAgent
from src.agents.preference_budget import PreferenceBudgetAgent
from src.agents.itinerary_planner import ItineraryPlannerAgent
from src.skills.poi_search import POISearchSkill


class Orchestrator:
    """
    Orchestrator：轻量编排器。
    根据意图类型调度 Agent，管理并行/串行执行。
    """
    
    def __init__(self):
        self.llm = get_llm_client()
        self.state_manager = StateManager()
        self.intent_agent = IntentRecognitionAgent(self.llm)
        self.preference_agent = PreferenceBudgetAgent(self.state_manager)
        self.planner_agent = ItineraryPlannerAgent()
        self.poi_search = POISearchSkill()
    
    async def process(self, session_id: str, user_input: str) -> ChatResponse:
        """处理用户输入的主入口"""
        
        # 1. 加载 State
        state = await self.state_manager.load(session_id)
        
        # 2. 意图识别
        intent_result = await self.intent_agent.recognize(
            user_input=user_input,
            conversation_history=state.recent_messages,
            current_state=state
        )
        
        # 保存用户消息
        await self.state_manager.add_message(session_id, "user", user_input)
        
        # 3. 根据意图分发
        if intent_result.intent == "generate_itinerary":
            return await self._handle_generate(session_id, state, intent_result)
        
        elif intent_result.intent == "update_preferences":
            return await self._handle_update_preferences(session_id, state, intent_result)
        
        elif intent_result.intent == "query_info":
            return await self._handle_qa(session_id, state, user_input)
        
        elif intent_result.intent == "confirm_itinerary":
            return await self._handle_confirm(session_id, state)
        
        elif intent_result.intent == "chitchat":
            return await self._handle_chitchat(session_id, state, user_input)
        
        else:
            return ChatResponse(
                session_id=session_id,
                response="抱歉，我没太明白你的意思。可以告诉我你想去哪里旅行吗？"
            )
    
    async def _handle_generate(
        self, session_id: str, state: TravelState, intent: IntentResult
    ) -> ChatResponse:
        """处理生成行程"""
        
        # 检查关键信息
        if intent.missing_required:
            questions = self._generate_questions(intent.missing_required)
            state.awaiting_user_input = True
            await self.state_manager.save(state)
            
            return ChatResponse(
                session_id=session_id,
                response=questions,
                preference_panel=self.preference_agent.get_panel_data(state.user_profile)
            )
        
        # 更新用户画像
        await self._update_profile_from_intent(session_id, state, intent)
        
        # 并行查询（POI + 偏好面板）
        profile = state.user_profile
        
        pois_task = self.poi_search.search_pois(
            city=profile.destination,
            keywords=profile.interests or [],
            top_n=30
        )
        
        panel_task = asyncio.to_thread(
            self.preference_agent.get_panel_data, profile
        )
        
        pois, preference_panel = await asyncio.gather(pois_task, panel_task)
        
        # 行程规划
        itinerary = self.planner_agent.plan(
            pois=pois,
            profile=profile
        )
        
        # 预算计算
        budget_panel = self.preference_agent.calculate_budget_breakdown(
            itinerary, profile
        )
        
        # 生成自然语言方案
        proposal_text = await self._generate_proposal(itinerary, profile, budget_panel)
        
        # 保存状态
        state.current_itinerary = itinerary
        state.current_budget = budget_panel
        state.current_preferences = preference_panel
        await self.state_manager.save(state)
        
        # 保存 Assistant 消息
        await self.state_manager.add_message(session_id, "assistant", proposal_text)
        
        return ChatResponse(
            session_id=session_id,
            response=proposal_text,
            intent="generate_itinerary",
            itinerary=itinerary,
            budget_panel=budget_panel,
            preference_panel=preference_panel
        )
    
    async def _handle_update_preferences(
        self, session_id: str, state: TravelState, intent: IntentResult
    ) -> ChatResponse:
        """处理偏好更新"""
        
        if not intent.preference_changes:
            return ChatResponse(
                session_id=session_id,
                response="好的，请告诉我你想更新什么偏好？"
            )
        
        # 更新偏好
        updated_profile = await self.preference_agent.update_preferences(
            session_id=session_id,
            changes=intent.preference_changes
        )
        
        # 如果有当前行程，重新规划
        if state.current_itinerary:
            # 简化版：重新生成
            # TODO: 增量更新
            pass
        
        preference_panel = self.preference_agent.get_panel_data(updated_profile)
        
        await self.state_manager.add_message(
            session_id, "assistant", 
            f"已更新你的偏好：{', '.join(c.field for c in intent.preference_changes)}"
        )
        
        return ChatResponse(
            session_id=session_id,
            response=f"已更新你的偏好！现在你的饮食偏好是：{', '.join(updated_profile.food_preferences or [])}",
            intent="update_preferences",
            preference_panel=preference_panel
        )
    
    async def _handle_qa(self, session_id: str, state: TravelState, user_input: str) -> ChatResponse:
        """处理问答"""
        # 简单直接调用 LLM 回答
        messages = [
            {"role": "system", "content": "你是一个旅行问答助手。回答用户关于旅行的问题。"},
            *[{"role": m.role, "content": m.content} for m in state.recent_messages[-5:]],
            {"role": "user", "content": user_input}
        ]
        
        answer = await self.llm.chat(messages, temperature=0.7)
        
        await self.state_manager.add_message(session_id, "assistant", answer)
        
        return ChatResponse(
            session_id=session_id,
            response=answer,
            intent="query_info"
        )
    
    async def _handle_confirm(self, session_id: str, state: TravelState) -> ChatResponse:
        """处理行程确认"""
        # TODO: 保存到数据库
        return ChatResponse(
            session_id=session_id,
            response="行程已确认！你可以在左侧"当前行程"查看详情。",
            intent="confirm_itinerary",
            itinerary=state.current_itinerary,
            budget_panel=state.current_budget,
            preference_panel=state.current_preferences
        )
    
    async def _handle_chitchat(self, session_id: str, state: TravelState, user_input: str) -> ChatResponse:
        """处理闲聊"""
        messages = [
            {"role": "system", "content": "你是一个友好的旅行助手，可以闲聊也可以帮用户规划旅行。"},
            {"role": "user", "content": user_input}
        ]
        
        response = await self.llm.chat(messages, temperature=0.8)
        await self.state_manager.add_message(session_id, "assistant", response)
        
        return ChatResponse(
            session_id=session_id,
            response=response,
            intent="chitchat"
        )
    
    async def _update_profile_from_intent(self, session_id: str, state: TravelState, intent: IntentResult):
        """从意图结果更新用户画像"""
        entities = intent.user_entities
        updates = {}
        
        for key in ["destination", "travel_days", "travel_dates", "travelers_count", 
                    "travelers_type", "budget_range", "food_preferences", 
                    "interests", "pace", "accommodation_preference", "special_requests"]:
            if key in entities and entities[key] is not None:
                updates[key] = entities[key]
        
        if updates:
            await self.state_manager.update_profile(session_id, **updates)
    
    def _generate_questions(self, missing_fields: List[str]) -> str:
        """生成追问"""
        field_names = {
            "destination": "目的地",
            "travel_days": "旅行天数",
            "travel_dates": "旅行日期"
        }
        
        questions = [f"{field_names.get(f, f)}是什么？" for f in missing_fields[:2]]
        return "为了给你规划行程，我还需要了解：" + "、".join(questions)
    
    async def _generate_proposal(
        self, itinerary: List[DayPlan], profile, budget: BudgetPanel
    ) -> str:
        """使用 LLM 生成自然语言方案"""
        # 构建行程描述
        itinerary_desc = "\n".join([
            f"Day {day.day_number}: " + " → ".join([a.poi_name for a in day.activities])
            for day in itinerary
        ])
        
        prompt = f"""根据以下行程，生成一段友好、自然的介绍文案，向用户介绍这个行程方案。

目的地：{profile.destination}
天数：{profile.travel_days} 天
预算：{profile.budget_range or '未设定'} 元
节奏：{profile.pace or '适中'}

行程概览：
{itinerary_desc}

预算概览：
- 总预算：{budget.total_budget or '未设定'}
- 预计花费：{budget.spent:.0f} 元
- 分项：住宿 {budget.breakdown.get('accommodation', 0):.0f}、餐饮 {budget.breakdown.get('meals', 0):.0f}、交通 {budget.breakdown.get('transport', 0):.0f}、门票 {budget.breakdown.get('tickets', 0):.0f}

要求：
1. 语气友好，像一个旅行顾问
2. 突出行程亮点
3. 提及预算情况
4. 告知用户可以点击"确认行程"或继续修改
5. 200-400字
"""
        
        return await self.llm.chat([{"role": "user", "content": prompt}], temperature=0.7)
```

### 4.10 FastAPI 主入口

**`backend/src/main.py`**

```python
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.core.database import init_db
from src.core.redis_client import get_redis, close_redis
from src.core.orchestrator import Orchestrator
from src.core.state_manager import StateManager
from src.core.schemas import WSMessage, ChatRequest, ChatResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    await init_db()
    await get_redis()
    print("应用启动完成")
    
    yield
    
    # 关闭时
    await close_redis()
    print("应用关闭")


app = FastAPI(
    title="Travel Agent API",
    description="AI-driven travel planning system",
    version="0.1.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局实例
orchestrator = Orchestrator()
state_manager = StateManager()


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """REST API：发送消息"""
    session_id = request.session_id or await state_manager.create_session()
    result = await orchestrator.process(session_id, request.message)
    return result


@app.post("/api/sessions")
async def create_session():
    """创建新会话"""
    session_id = await state_manager.create_session()
    return {"session_id": session_id}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """获取会话状态"""
    state = await state_manager.load(session_id)
    return {
        "session_id": session_id,
        "profile": state.user_profile.model_dump(),
        "has_itinerary": state.current_itinerary is not None
    }


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket：实时对话"""
    await websocket.accept()
    session_id = None
    
    try:
        while True:
            # 接收消息
            raw_data = await websocket.receive_text()
            data = json.loads(raw_data)
            
            message_type = data.get("type", "user_message")
            
            if message_type == "user_message":
                session_id = data.get("session_id") or await state_manager.create_session()
                user_message = data.get("content", "")
                
                # 发送意图识别开始（可选）
                await websocket.send_json({
                    "type": "intent_recognized",
                    "session_id": session_id,
                    "metadata": {"status": "processing"}
                })
                
                # 处理消息
                result = await orchestrator.process(session_id, user_message)
                
                # 发送 Assistant 回复
                await websocket.send_json({
                    "type": "assistant_message",
                    "session_id": session_id,
                    "content": result.response,
                    "metadata": {
                        "intent": result.intent,
                        "has_itinerary": result.itinerary is not None
                    }
                })
                
                # 如果有行程，发送行程更新
                if result.itinerary:
                    await websocket.send_json({
                        "type": "itinerary_update",
                        "session_id": session_id,
                        "data": [day.model_dump() for day in result.itinerary]
                    })
                
                # 发送预算更新
                if result.budget_panel:
                    await websocket.send_json({
                        "type": "budget_update",
                        "session_id": session_id,
                        "data": result.budget_panel.model_dump()
                    })
                
                # 发送偏好更新
                if result.preference_panel:
                    await websocket.send_json({
                        "type": "preference_update",
                        "session_id": session_id,
                        "data": result.preference_panel.model_dump()
                    })
            
            elif message_type == "confirm_itinerary":
                if session_id:
                    result = await orchestrator.process(session_id, "确认行程")
                    await websocket.send_json({
                        "type": "assistant_message",
                        "session_id": session_id,
                        "content": result.response
                    })
    
    except WebSocketDisconnect:
        print(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.send_json({
            "type": "error",
            "content": f"处理出错: {str(e)}"
        })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
```

---

## 五、前端技术方案

### 5.1 Zustand 全局状态管理

**`frontend/src/stores/appStore.ts`**

```typescript
import { create } from 'zustand';

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  displayType?: 'text' | 'itinerary' | 'budget_update' | 'preference_update';
}

export interface Activity {
  poi_name: string;
  category: string;
  start_time?: string;
  end_time?: string;
  duration_min: number;
  ticket_price?: number;
  recommendation_reason?: string;
}

export interface DayPlan {
  day_number: number;
  date?: string;
  theme?: string;
  activities: Activity[];
  total_cost: number;
}

export interface BudgetPanel {
  total_budget?: number;
  spent: number;
  remaining?: number;
  breakdown: Record<string, number>;
  status: 'within_budget' | 'over_budget' | 'unknown';
}

export interface PreferencePanel {
  destination?: string;
  travel_days?: number;
  travel_dates?: string;
  travelers_count?: number;
  travelers_type?: string;
  budget_range?: number;
  food_preferences: string[];
  interests: string[];
  pace?: string;
  special_requests: string[];
}

export interface Session {
  id: string;
  title: string;
  destination?: string;
  createdAt: string;
}

interface AppState {
  // 会话
  currentSessionId: string | null;
  sessions: Session[];
  
  // 对话
  messages: Message[];
  isLoading: boolean;
  
  // 行程
  currentItinerary: DayPlan[] | null;
  
  // 面板
  budgetPanel: BudgetPanel | null;
  preferencePanel: PreferencePanel | null;
  
  // 视图
  activeView: 'chat' | 'itinerary';
  
  // Actions
  setCurrentSession: (id: string | null) => void;
  addSession: (session: Session) => void;
  addMessage: (message: Message) => void;
  setMessages: (messages: Message[]) => void;
  setIsLoading: (loading: boolean) => void;
  setCurrentItinerary: (itinerary: DayPlan[] | null) => void;
  setBudgetPanel: (panel: BudgetPanel | null) => void;
  setPreferencePanel: (panel: PreferencePanel | null) => void;
  setActiveView: (view: 'chat' | 'itinerary') => void;
  clearCurrentSession: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  currentSessionId: null,
  sessions: [],
  messages: [],
  isLoading: false,
  currentItinerary: null,
  budgetPanel: null,
  preferencePanel: null,
  activeView: 'chat',
  
  setCurrentSession: (id) => set({ currentSessionId: id }),
  
  addSession: (session) => set((state) => ({
    sessions: [session, ...state.sessions],
    currentSessionId: session.id
  })),
  
  addMessage: (message) => set((state) => ({
    messages: [...state.messages, message]
  })),
  
  setMessages: (messages) => set({ messages }),
  
  setIsLoading: (loading) => set({ isLoading: loading }),
  
  setCurrentItinerary: (itinerary) => set({ currentItinerary: itinerary }),
  
  setBudgetPanel: (panel) => set({ budgetPanel: panel }),
  
  setPreferencePanel: (panel) => set({ preferencePanel: panel }),
  
  setActiveView: (view) => set({ activeView: view }),
  
  clearCurrentSession: () => set({
    currentSessionId: null,
    messages: [],
    currentItinerary: null,
    budgetPanel: null,
    preferencePanel: null,
    activeView: 'chat'
  })
}));
```

### 5.2 WebSocket Hook

**`frontend/src/hooks/useWebSocket.ts`**

```typescript
import { useEffect, useRef, useCallback } from 'react';
import { useAppStore } from '@/stores/appStore';

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws/chat';

export function useWebSocket() {
  const ws = useRef<WebSocket | null>(null);
  const store = useAppStore();
  
  const connect = useCallback(() => {
    if (ws.current?.readyState === WebSocket.OPEN) return;
    
    ws.current = new WebSocket(WS_URL);
    
    ws.current.onopen = () => {
      console.log('WebSocket connected');
    };
    
    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleMessage(data);
    };
    
    ws.current.onclose = () => {
      console.log('WebSocket disconnected');
    };
    
    ws.current.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }, []);
  
  const handleMessage = useCallback((data: any) => {
    switch (data.type) {
      case 'assistant_message':
        store.addMessage({
          id: Date.now().toString(),
          role: 'assistant',
          content: data.content,
          timestamp: new Date().toISOString()
        });
        store.setIsLoading(false);
        break;
        
      case 'itinerary_update':
        store.setCurrentItinerary(data.data);
        break;
        
      case 'budget_update':
        store.setBudgetPanel(data.data);
        break;
        
      case 'preference_update':
        store.setPreferencePanel(data.data);
        break;
        
      case 'error':
        store.addMessage({
          id: Date.now().toString(),
          role: 'assistant',
          content: `❌ ${data.content}`,
          timestamp: new Date().toISOString()
        });
        store.setIsLoading(false);
        break;
    }
  }, [store]);
  
  const sendMessage = useCallback((content: string, sessionId?: string) => {
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) {
      connect();
    }
    
    store.addMessage({
      id: Date.now().toString(),
      role: 'user',
      content,
      timestamp: new Date().toISOString()
    });
    
    store.setIsLoading(true);
    
    ws.current?.send(JSON.stringify({
      type: 'user_message',
      content,
      session_id: sessionId
    }));
  }, [connect, store]);
  
  const confirmItinerary = useCallback((sessionId: string) => {
    ws.current?.send(JSON.stringify({
      type: 'confirm_itinerary',
      session_id: sessionId
    }));
  }, []);
  
  useEffect(() => {
    connect();
    return () => {
      ws.current?.close();
    };
  }, [connect]);
  
  return { sendMessage, confirmItinerary, isConnected: !!ws.current };
}
```

### 5.3 主页面布局

**`frontend/src/app/page.tsx`**

```tsx
'use client';

import { LeftSidebar } from '@/components/layout/LeftSidebar';
import { MainContent } from '@/components/layout/MainContent';
import { RightPanel } from '@/components/layout/RightPanel';

export default function Home() {
  return (
    <div className="flex h-screen bg-gray-50">
      {/* 左侧栏 */}
      <LeftSidebar />
      
      {/* 中间栏 */}
      <MainContent />
      
      {/* 右侧栏 */}
      <RightPanel />
    </div>
  );
}
```

### 5.4 左侧栏组件

**`frontend/src/components/layout/LeftSidebar.tsx`**

```tsx
'use client';

import { Plus, MessageSquare, MapPin, History } from 'lucide-react';
import { useAppStore } from '@/stores/appStore';
import { v4 as uuidv4 } from 'uuid';

export function LeftSidebar() {
  const store = useAppStore();
  
  const handleNewChat = () => {
    const newSession = {
      id: uuidv4(),
      title: '新对话',
      createdAt: new Date().toISOString()
    };
    store.addSession(newSession);
    store.clearCurrentSession();
  };
  
  return (
    <div className="w-64 bg-white border-r border-gray-200 flex flex-col">
      {/* Logo */}
      <div className="p-4 border-b border-gray-200">
        <h1 className="text-xl font-bold text-gray-800">旅行 Agent</h1>
      </div>
      
      {/* 新建对话 */}
      <div className="p-3">
        <button
          onClick={handleNewChat}
          className="w-full flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
        >
          <Plus size={18} />
          <span>新建对话</span>
        </button>
      </div>
      
      {/* 对话列表 */}
      <div className="flex-1 overflow-y-auto">
        <div className="px-3 py-2 text-xs font-semibold text-gray-500 uppercase">
          对话列表
        </div>
        {store.sessions.map((session) => (
          <button
            key={session.id}
            onClick={() => store.setCurrentSession(session.id)}
            className={`w-full flex items-center gap-2 px-3 py-2 mx-2 rounded-lg text-left transition ${
              store.currentSessionId === session.id
                ? 'bg-blue-50 text-blue-700'
                : 'text-gray-700 hover:bg-gray-100'
            }`}
          >
            <MessageSquare size={16} />
            <span className="truncate text-sm">{session.title}</span>
          </button>
        ))}
      </div>
      
      {/* 当前行程 */}
      <div className="border-t border-gray-200">
        <button
          onClick={() => store.setActiveView('itinerary')}
          className={`w-full flex items-center gap-2 px-4 py-3 text-left transition ${
            store.activeView === 'itinerary'
              ? 'bg-blue-50 text-blue-700'
              : 'text-gray-700 hover:bg-gray-100'
          }`}
        >
          <MapPin size={18} />
          <span className="font-medium">当前行程</span>
        </button>
      </div>
      
      {/* 历史行程 */}
      <div className="border-t border-gray-200">
        <div className="px-4 py-2 text-xs font-semibold text-gray-500 uppercase">
          历史行程
        </div>
        <button className="w-full flex items-center gap-2 px-4 py-2 text-gray-700 hover:bg-gray-100 text-left">
          <History size={16} />
          <span className="text-sm">成都 4日游</span>
        </button>
      </div>
    </div>
  );
}
```

### 5.5 中间栏组件

**`frontend/src/components/layout/MainContent.tsx`**

```tsx
'use client';

import { useAppStore } from '@/stores/appStore';
import { ChatInterface } from '@/components/chat/ChatInterface';
import { ItineraryView } from '@/components/itinerary/ItineraryView';

export function MainContent() {
  const activeView = useAppStore((state) => state.activeView);
  
  return (
    <div className="flex-1 flex flex-col min-w-0">
      {activeView === 'chat' ? <ChatInterface /> : <ItineraryView />}
    </div>
  );
}
```

**`frontend/src/components/chat/ChatInterface.tsx`**

```tsx
'use client';

import { useState, useRef, useEffect } from 'react';
import { Send } from 'lucide-react';
import { useAppStore } from '@/stores/appStore';
import { useWebSocket } from '@/hooks/useWebSocket';
import { ItineraryCard } from './ItineraryCard';

export function ChatInterface() {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const store = useAppStore();
  const { sendMessage } = useWebSocket();
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || store.isLoading) return;
    
    sendMessage(input.trim(), store.currentSessionId || undefined);
    setInput('');
  };
  
  // 自动滚动到底部
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [store.messages]);
  
  return (
    <div className="flex-1 flex flex-col">
      {/* 消息列表 */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {store.messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-gray-400">
            <h2 className="text-2xl font-bold mb-2">想去哪里旅行？</h2>
            <p>告诉我目的地和天数，我来帮你规划</p>
          </div>
        )}
        
        {store.messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                msg.role === 'user'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white border border-gray-200 text-gray-800'
              }`}
            >
              <p className="whitespace-pre-wrap">{msg.content}</p>
              
              {/* 如果是行程消息，显示行程卡片 */}
              {msg.displayType === 'itinerary' && store.currentItinerary && (
                <ItineraryCard itinerary={store.currentItinerary} />
              )}
            </div>
          </div>
        ))}
        
        {store.isLoading && (
          <div className="flex justify-start">
            <div className="bg-white border border-gray-200 rounded-2xl px-4 py-3">
              <div className="flex gap-1">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-100" />
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-200" />
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>
      
      {/* 输入框 */}
      <form onSubmit={handleSubmit} className="p-4 border-t border-gray-200 bg-white">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="输入你想去的城市和天数..."
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            type="submit"
            disabled={!input.trim() || store.isLoading}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
          >
            <Send size={18} />
          </button>
        </div>
      </form>
    </div>
  );
}
```

### 5.6 右侧面板

**`frontend/src/components/layout/RightPanel.tsx`**

```tsx
'use client';

import { useAppStore } from '@/stores/appStore';
import { BudgetPanel } from '@/components/panels/BudgetPanel';
import { PreferencePanel } from '@/components/panels/PreferencePanel';

export function RightPanel() {
  const budgetPanel = useAppStore((state) => state.budgetPanel);
  const preferencePanel = useAppStore((state) => state.preferencePanel);
  
  return (
    <div className="w-80 bg-white border-l border-gray-200 flex flex-col overflow-y-auto">
      {/* 预算面板 */}
      {budgetPanel && <BudgetPanel data={budgetPanel} />}
      
      {/* 分隔线 */}
      <div className="border-t border-gray-200" />
      
      {/* 偏好面板 */}
      {preferencePanel && <PreferencePanel data={preferencePanel} />}
      
      {/* 空状态 */}
      {!budgetPanel && !preferencePanel && (
        <div className="flex-1 flex flex-col items-center justify-center text-gray-400 p-8">
          <p className="text-center">开始对话后，你的偏好和预算信息将显示在这里</p>
        </div>
      )}
    </div>
  );
}
```

**`frontend/src/components/panels/BudgetPanel.tsx`**

```tsx
'use client';

import { Wallet } from 'lucide-react';
import { BudgetPanel as BudgetPanelType } from '@/stores/appStore';

interface Props {
  data: BudgetPanelType;
}

export function BudgetPanel({ data }: Props) {
  const percentage = data.total_budget
    ? Math.min(100, (data.spent / data.total_budget) * 100)
    : 0;
  
  return (
    <div className="p-4">
      <div className="flex items-center gap-2 mb-4">
        <Wallet size={18} className="text-blue-600" />
        <h3 className="font-semibold text-gray-800">预算</h3>
      </div>
      
      <div className="space-y-3">
        <div className="flex justify-between text-sm">
          <span className="text-gray-500">总预算</span>
          <span className="font-medium">
            {data.total_budget ? `¥${data.total_budget.toLocaleString()}` : '未设定'}
          </span>
        </div>
        
        <div className="flex justify-between text-sm">
          <span className="text-gray-500">预计花费</span>
          <span className="font-medium text-orange-600">
            ¥{data.spent.toLocaleString()}
          </span>
        </div>
        
        {data.remaining !== undefined && (
          <div className="flex justify-between text-sm">
            <span className="text-gray-500">剩余</span>
            <span className={`font-medium ${data.remaining < 0 ? 'text-red-600' : 'text-green-600'}`}>
              ¥{data.remaining.toLocaleString()}
            </span>
          </div>
        )}
        
        {/* 进度条 */}
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className={`h-2 rounded-full transition-all ${
              percentage > 90 ? 'bg-red-500' : percentage > 70 ? 'bg-yellow-500' : 'bg-green-500'
            }`}
            style={{ width: `${percentage}%` }}
          />
        </div>
        
        {/* 分项明细 */}
        <div className="pt-2 border-t border-gray-100">
          <p className="text-xs font-medium text-gray-500 mb-2">分项明细</p>
          {Object.entries(data.breakdown).map(([key, value]) => (
            <div key={key} className="flex justify-between text-xs py-1">
              <span className="text-gray-500 capitalize">{key}</span>
              <span>¥{value.toLocaleString()}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
```

**`frontend/src/components/panels/PreferencePanel.tsx`**

```tsx
'use client';

import { MapPin, Calendar, Users, DollarSign, Heart, Utensils, Footprints } from 'lucide-react';
import { PreferencePanel as PreferencePanelType } from '@/stores/appStore';

interface Props {
  data: PreferencePanelType;
}

export function PreferencePanel({ data }: Props) {
  return (
    <div className="p-4">
      <div className="flex items-center gap-2 mb-4">
        <Heart size={18} className="text-pink-500" />
        <h3 className="font-semibold text-gray-800">偏好</h3>
      </div>
      
      <div className="space-y-3">
        {/* 基础信息 */}
        {data.destination && (
          <div className="flex items-center gap-2 text-sm">
            <MapPin size={14} className="text-gray-400" />
            <span className="text-gray-500">目的地：</span>
            <span className="font-medium">{data.destination}</span>
          </div>
        )}
        
        {data.travel_days && (
          <div className="flex items-center gap-2 text-sm">
            <Calendar size={14} className="text-gray-400" />
            <span className="text-gray-500">天数：</span>
            <span className="font-medium">{data.travel_days} 天</span>
          </div>
        )}
        
        {data.travelers_count && (
          <div className="flex items-center gap-2 text-sm">
            <Users size={14} className="text-gray-400" />
            <span className="text-gray-500">人数：</span>
            <span className="font-medium">{data.travelers_count} 人</span>
            {data.travelers_type && <span className="text-gray-400">({data.travelers_type})</span>}
          </div>
        )}
        
        {data.budget_range && (
          <div className="flex items-center gap-2 text-sm">
            <DollarSign size={14} className="text-gray-400" />
            <span className="text-gray-500">预算：</span>
            <span className="font-medium">¥{data.budget_range.toLocaleString()}</span>
          </div>
        )}
        
        {/* 饮食偏好 */}
        {data.food_preferences.length > 0 && (
          <div className="pt-2 border-t border-gray-100">
            <div className="flex items-center gap-2 text-sm mb-2">
              <Utensils size={14} className="text-gray-400" />
              <span className="text-gray-500">饮食偏好</span>
            </div>
            <div className="flex flex-wrap gap-1">
              {data.food_preferences.map((food) => (
                <span
                  key={food}
                  className="px-2 py-1 bg-orange-50 text-orange-700 text-xs rounded-full"
                >
                  {food}
                </span>
              ))}
            </div>
          </div>
        )}
        
        {/* 兴趣爱好 */}
        {data.interests.length > 0 && (
          <div className="pt-2 border-t border-gray-100">
            <div className="flex items-center gap-2 text-sm mb-2">
              <Heart size={14} className="text-gray-400" />
              <span className="text-gray-500">兴趣爱好</span>
            </div>
            <div className="flex flex-wrap gap-1">
              {data.interests.map((interest) => (
                <span
                  key={interest}
                  className="px-2 py-1 bg-blue-50 text-blue-700 text-xs rounded-full"
                >
                  {interest}
                </span>
              ))}
            </div>
          </div>
        )}
        
        {/* 节奏 */}
        {data.pace && (
          <div className="flex items-center gap-2 text-sm pt-2 border-t border-gray-100">
            <Footprints size={14} className="text-gray-400" />
            <span className="text-gray-500">节奏：</span>
            <span className="font-medium">
              {data.pace === 'relaxed' ? '轻松' : data.pace === 'moderate' ? '适中' : '紧凑'}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
```

### 5.7 行程视图

**`frontend/src/components/itinerary/ItineraryView.tsx`**

```tsx
'use client';

import { useAppStore } from '@/stores/appStore';
import { Check } from 'lucide-react';
import { useWebSocket } from '@/hooks/useWebSocket';

export function ItineraryView() {
  const itinerary = useAppStore((state) => state.currentItinerary);
  const sessionId = useAppStore((state) => state.currentSessionId);
  const { confirmItinerary } = useWebSocket();
  
  if (!itinerary) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-gray-400">
        <p>还没有行程，先开始对话规划吧</p>
      </div>
    );
  }
  
  return (
    <div className="flex-1 overflow-y-auto p-6">
      <div className="max-w-3xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-gray-800">行程详情</h2>
          <button
            onClick={() => sessionId && confirmItinerary(sessionId)}
            className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition"
          >
            <Check size={18} />
            确认行程
          </button>
        </div>
        
        {itinerary.map((day) => (
          <div key={day.day_number} className="mb-8">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold">
                {day.day_number}
              </div>
              <div>
                <h3 className="font-semibold text-gray-800">第 {day.day_number} 天</h3>
                {day.theme && <p className="text-sm text-gray-500">{day.theme}</p>}
              </div>
            </div>
            
            <div className="ml-5 pl-8 border-l-2 border-blue-200 space-y-4">
              {day.activities.map((activity, idx) => (
                <div key={idx} className="bg-white rounded-lg p-4 border border-gray-200 shadow-sm">
                  <div className="flex items-start justify-between">
                    <div>
                      <p className="font-medium text-gray-800">{activity.poi_name}</p>
                      <p className="text-sm text-gray-500">{activity.category}</p>
                      {activity.recommendation_reason && (
                        <p className="text-sm text-gray-600 mt-1">{activity.recommendation_reason}</p>
                      )}
                    </div>
                    <div className="text-right text-sm text-gray-500">
                      {activity.start_time && (
                        <p>{activity.start_time} - {activity.end_time}</p>
                      )}
                      <p>{activity.duration_min} 分钟</p>
                      {activity.ticket_price && (
                        <p className="text-orange-600">¥{activity.ticket_price}</p>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
            
            <div className="ml-5 pl-8 mt-2 text-sm text-gray-500">
              当日花费：¥{day.total_cost.toLocaleString()}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
```

---

## 六、实现路线图（Step by Step）

### Phase 1: 基础框架搭建（Week 1，第 1-3 天）

**Step 1: 项目初始化**
```bash
# 1. 创建目录
mkdir travel-agent && cd travel-agent
mkdir -p backend/src backend/tests
npx create-next-app@latest frontend --typescript --tailwind --eslint --app --src-dir --import-alias "@/*"

# 2. 后端依赖
cd backend
pip install fastapi uvicorn pydantic sqlalchemy asyncpg redis openai aiohttp beautifulsoup4 duckduckgo-search numpy scikit-learn python-dotenv pytest pytest-asyncio

# 3. 前端依赖
cd ../frontend
npm install zustand axios lucide-react

# 4. 创建文件结构
cd ../backend
mkdir -p src/{agents,core,skills,planner,memory,api/{routes,middleware},config}
touch src/__init__.py
touch src/agents/__init__.py src/core/__init__.py src/skills/__init__.py
```

**Step 2: 数据库和配置**
```bash
# 1. 启动 Docker Compose
docker-compose up -d postgres redis

# 2. 依次创建以下文件（按顺序）：
# - backend/src/config/settings.py
# - backend/src/core/models.py
# - backend/src/core/schemas.py
# - backend/src/core/database.py
# - backend/src/core/redis_client.py

# 3. 初始化数据库表
# 运行 init_db() 或执行 schema.sql
```

**Step 3: FastAPI 基础框架**
```bash
# 创建：
# - backend/src/main.py（基础路由 + health check）
# - backend/src/api/routes/chat.py
# - backend/Dockerfile
# - docker-compose.yml

# 验证：
curl http://localhost:8000/health
# 期望返回：{"status": "ok"}
```

**验收标准**：
- [ ] Docker Compose 能启动 postgres + redis + backend
- [ ] `/health` 接口返回 ok
- [ ] 数据库表已创建

---

### Phase 2: 核心后端实现（Week 1-2，第 4-8 天）

**Step 4: LLM 客户端**
```bash
# 创建 backend/src/core/llm_client.py
# 功能：
# - chat() 普通对话
# - structured_call() 结构化 JSON 输出
# - 全局单例 get_llm_client()

# 验证：
python -c "from src.core.llm_client import get_llm_client; print('LLM client OK')"
```

**Step 5: Skill 层**
```bash
# 按顺序创建：
# 1. backend/src/skills/web_search.py（DuckDuckGo 搜索）
# 2. backend/src/skills/web_crawler.py（异步网页爬取）
# 3. backend/src/skills/poi_search.py（POI 搜索整合）

# 验证：
python -c "
import asyncio
from src.skills.web_search import WebSearchSkill
skill = WebSearchSkill()
results = asyncio.run(skill.search('成都 景点'))
print(f'找到 {len(results)} 个结果')
"
```

**Step 6: 意图识别 Agent**
```bash
# 创建 backend/src/agents/intent_recognition.py
# 功能：
# - recognize() 识别意图
# - _detect_preference_changes() 检测偏好变更

# 验证：
python -c "
import asyncio
from src.core.llm_client import get_llm_client
from src.agents.intent_recognition import IntentRecognitionAgent
from src.core.schemas import TravelState, UserProfile

llm = get_llm_client()
agent = IntentRecognitionAgent(llm)
state = TravelState(session_id='test', user_profile=UserProfile())
result = asyncio.run(agent.recognize('我想去成都玩4天', [], state))
print(f'意图: {result.intent}')
print(f'实体: {result.user_entities}')
"
```

**Step 7: 行程规划 Agent**
```bash
# 创建 backend/src/agents/itinerary_planner.py
# 功能：
# - plan() 生成行程
# - _score_pois() POI 打分
# - _cluster_pois() 区域聚类（DBSCAN）
# - _assign_to_days() 日分配
# - _build_schedule() 时间表生成

# 验证：
python -c "
from src.agents.itinerary_planner import ItineraryPlannerAgent
from src.core.schemas import UserProfile

agent = ItineraryPlannerAgent()
pois = [...]  # 测试数据
itinerary = agent.plan(pois, UserProfile(destination='成都', travel_days=3))
print(f'生成了 {len(itinerary)} 天的行程')
"
```

**Step 8: Orchestrator + API**
```bash
# 创建 backend/src/core/orchestrator.py
# 更新 backend/src/main.py（添加 WebSocket + REST API）

# 验证（端到端）：
curl -X POST http://localhost:8000/api/sessions
# 返回 session_id

curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "我想去成都玩3天"}'
# 返回行程数据
```

**验收标准**：
- [ ] LLM 能正确识别意图和提取实体
- [ ] 爬虫能搜索并提取 POI 信息
- [ ] 能生成结构化行程（JSON）
- [ ] REST API 返回完整响应

---

### Phase 3: 前端实现（Week 2，第 9-12 天）

**Step 9: 前端基础结构**
```bash
# 1. 创建 stores、hooks、components 目录
# 2. 创建 appStore.ts（Zustand 状态管理）
# 3. 创建 useWebSocket.ts（WebSocket hook）

# 验证：
cd frontend
npm run dev
# 访问 http://localhost:3000
```

**Step 10: 三栏布局**
```bash
# 按顺序创建组件：
# 1. LeftSidebar.tsx（左侧栏）
# 2. MainContent.tsx（中间栏容器）
# 3. RightPanel.tsx（右侧栏容器）
# 4. ChatInterface.tsx（对话界面）
# 5. BudgetPanel.tsx（预算面板）
# 6. PreferencePanel.tsx（偏好面板）

# 更新 page.tsx

# 验证：
# - 能看到三栏布局
# - 能输入消息并发送
```

**Step 11: 行程展示**
```bash
# 创建：
# - ItineraryView.tsx（行程时间线视图）
# - ItineraryCard.tsx（行程卡片）
# - DayPlanCard.tsx（每日计划）

# 验证：
# - 后端返回行程后，中间栏显示时间线
# - 点击"当前行程"能查看详情
```

**Step 12: 行程确认流程**
```bash
# 更新：
# - ItineraryView.tsx 添加确认按钮
# - useWebSocket.ts 添加 confirmItinerary
# - 后端 WebSocket 处理 confirm_itinerary 消息

# 验证：
# - 能点击"确认行程"
# - 确认后左侧"当前行程"可点击
```

**验收标准**：
- [ ] 三栏布局正常显示
- [ ] 能输入消息，后端返回行程
- [ ] 右侧面板实时更新预算和偏好
- [ ] 行程时间线正确展示
- [ ] 能确认行程

---

### Phase 4: 集成测试与优化（Week 3，第 13-15 天）

**Step 13: 端到端测试**
```bash
# 测试场景：
# 1. "我想去成都玩4天，预算3000"
#    - 期望：提取 destination=成都, travel_days=4, budget=3000
#    - 期望：生成4天行程，预算面板显示
# 2. "再加个甜品爱好"
#    - 期望：偏好面板更新，food_preferences 增加"甜品"
# 3. "确认行程"
#    - 期望：行程保存，显示确认成功
```

**Step 14: 性能优化**
```bash
# 1. 添加缓存（Redis）
# 2. 优化爬虫并发数
# 3. 添加请求超时处理
# 4. 添加错误降级（爬虫失败时使用本地数据）
```

**Step 15: Bug 修复与文档**
```bash
# 1. 修复测试中发现的问题
# 2. 更新 README.md
# 3. 编写部署文档
```

**验收标准**：
- [ ] 完整用户流程能跑通
- [ ] 对话自然，追问合理
- [ ] 行程展示清晰
- [ ] 预算估算合理

---

## 七、API 接口规范

### 7.1 REST API

```yaml
openapi: 3.0.0
info:
  title: Travel Agent API
  version: 0.1.0

paths:
  /health:
    get:
      summary: 健康检查
      responses:
        200:
          description: OK
          content:
            application/json:
              schema:
                type: object
                properties:
                  status: { type: string, example: "ok" }

  /api/sessions:
    post:
      summary: 创建新会话
      responses:
        200:
          description: 创建成功
          content:
            application/json:
              schema:
                type: object
                properties:
                  session_id: { type: string }

  /api/sessions/{session_id}:
    get:
      summary: 获取会话状态
      parameters:
        - name: session_id
          in: path
          required: true
          schema: { type: string }
      responses:
        200:
          description: 会话状态
          content:
            application/json:
              schema:
                type: object
                properties:
                  session_id: { type: string }
                  profile: { type: object }
                  has_itinerary: { type: boolean }

  /api/chat:
    post:
      summary: 发送消息（REST）
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                session_id: { type: string, nullable: true }
                message: { type: string }
      responses:
        200:
          description: 处理结果
          content:
            application/json:
              schema:
                type: object
                properties:
                  session_id: { type: string }
                  response: { type: string }
                  intent: { type: string }
                  itinerary: { type: array }
                  budget_panel: { type: object }
                  preference_panel: { type: object }
```

### 7.2 WebSocket API

```yaml
WebSocket: ws://localhost:8000/ws/chat

# 客户端 → 服务端
messages:
  user_message:
    type: "user_message"
    content: "我想去成都玩4天"
    session_id: "uuid-or-null"

  confirm_itinerary:
    type: "confirm_itinerary"
    session_id: "uuid"

# 服务端 → 客户端
messages:
  assistant_message:
    type: "assistant_message"
    session_id: "uuid"
    content: "行程已生成..."
    metadata:
      intent: "generate_itinerary"
      has_itinerary: true

  itinerary_update:
    type: "itinerary_update"
    session_id: "uuid"
    data: [DayPlan]

  budget_update:
    type: "budget_update"
    session_id: "uuid"
    data: BudgetPanel

  preference_update:
    type: "preference_update"
    session_id: "uuid"
    data: PreferencePanel

  error:
    type: "error"
    content: "错误信息"
```

---

## 八、部署方案

### 8.1 本地开发部署

```bash
# 1. 克隆代码
git clone <repo-url>
cd travel-agent

# 2. 配置环境变量
cp backend/.env.example backend/.env
# 编辑 .env，填入 LLM API Key

# 3. 启动服务
docker-compose up -d

# 4. 验证
# 后端: http://localhost:8000
# 前端: http://localhost:3000
# 数据库: localhost:5432
# Redis: localhost:6379
```

### 8.2 生产部署（Railway / Render）

```yaml
# railway.yml
services:
  - name: travel-agent-backend
    source: ./backend
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn src.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: DATABASE_URL
        fromDatabase:
          name: travel-agent-db
          property: connectionString
      - key: REDIS_URL
        fromService:
          name: travel-agent-redis
          type: redis
          property: connectionString

  - name: travel-agent-frontend
    source: ./frontend
    buildCommand: npm install && npm run build
    startCommand: npm start
    envVars:
      - key: NEXT_PUBLIC_API_URL
        value: https://travel-agent-backend.up.railway.app
```

### 8.3 环境变量清单

| 变量名 | 说明 | 必填 | 示例 |
|--------|------|------|------|
| DATABASE_URL | PostgreSQL 连接串 | 是 | `postgresql+asyncpg://...` |
| REDIS_URL | Redis 连接串 | 是 | `redis://localhost:6379/0` |
| LLM_API_KEY | LLM API 密钥 | 是 | `sk-...` |
| LLM_BASE_URL | LLM API 基础地址 | 否 | `https://api.anthropic.com/v1` |
| LLM_MODEL | 模型名称 | 否 | `claude-sonnet-4-6` |
| APP_ENV | 运行环境 | 否 | `development` |
| DEBUG | 调试模式 | 否 | `true` |

---

## 附录：文件清单

### 后端文件（共 20+ 个）

```
backend/
├── src/
│   ├── main.py                           # FastAPI 入口
│   ├── config/
│   │   ├── settings.py                   # 配置管理
│   │   └── schema.sql                    # 数据库建表语句
│   ├── core/
│   │   ├── __init__.py
│   │   ├── models.py                     # SQLAlchemy ORM 模型
│   │   ├── schemas.py                    # Pydantic 数据模型
│   │   ├── database.py                   # 数据库连接
│   │   ├── redis_client.py               # Redis 连接
│   │   ├── llm_client.py                 # LLM 客户端
│   │   ├── state_manager.py              # State 管理
│   │   ├── event_bus.py                  # 事件总线
│   │   └── orchestrator.py               # Orchestrator 编排器
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── intent_recognition.py         # 意图识别 Agent
│   │   ├── preference_budget.py          # 偏好预算 Agent
│   │   └── itinerary_planner.py          # 行程规划 Agent
│   ├── skills/
│   │   ├── __init__.py
│   │   ├── web_search.py                 # 网络搜索
│   │   ├── web_crawler.py                # 网页爬取
│   │   └── poi_search.py                 # POI 搜索
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       └── chat.py                   # 聊天路由
│   └── planner/
│       └── __init__.py
├── tests/
│   ├── unit/
│   └── integration/
├── Dockerfile
├── requirements.txt / pyproject.toml
└── .env
```

### 前端文件（共 15+ 个）

```
frontend/
├── src/
│   ├── app/
│   │   ├── page.tsx                      # 主页面（三栏布局）
│   │   ├── layout.tsx
│   │   └── globals.css
│   ├── components/
│   │   ├── layout/
│   │   │   ├── LeftSidebar.tsx           # 左侧栏
│   │   │   ├── MainContent.tsx           # 中间栏容器
│   │   │   └── RightPanel.tsx            # 右侧栏容器
│   │   ├── chat/
│   │   │   ├── ChatInterface.tsx         # 对话界面
│   │   │   └── ItineraryCard.tsx         # 行程卡片
│   │   ├── itinerary/
│   │   │   └── ItineraryView.tsx         # 行程视图
│   │   └── panels/
│   │       ├── BudgetPanel.tsx           # 预算面板
│   │       └── PreferencePanel.tsx       # 偏好面板
│   ├── hooks/
│   │   └── useWebSocket.ts               # WebSocket Hook
│   ├── stores/
│   │   └── appStore.ts                   # Zustand 全局状态
│   ├── types/
│   │   └── index.ts                      # TypeScript 类型
│   └── lib/
│       └── api.ts                        # API 客户端
├── package.json
├── next.config.js
└── Dockerfile
```

---

**文档结束**

> 本技术方案基于 v4 设计文档，面向 MVP Phase 1（单轮单城市行程生成）。按此方案执行，预计 3 周可完成核心功能。每个 Step 都有明确的验收标准，可逐步验证。
