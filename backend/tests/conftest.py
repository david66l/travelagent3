"""Global test fixtures and monkeypatches."""

import pytest
from unittest.mock import AsyncMock


# ===== 1. LLM Global Mock (module-level, before any imports) =====
# Must set BEFORE agents modules are imported, since they do
# `from core.llm_client import llm` which binds a local reference.

_llm_mock = AsyncMock()
_llm_mock.chat = AsyncMock(return_value="")
_llm_mock.structured_call = AsyncMock(return_value=None)
_llm_mock.json_chat = AsyncMock(return_value={})

# Patch the source module
import core.llm_client
core.llm_client.llm = _llm_mock

# Patch every module that already imported llm locally
for _mod_name in [
    "agents.information_collection",
    "agents.intent_recognition",
    "agents.itinerary_planner",
    "agents.proposal_generation",
    "agents.qa_agent",
    "skills.poi_search",
]:
    try:
        _mod = __import__(_mod_name, fromlist=["llm"])
        if hasattr(_mod, "llm"):
            _mod.llm = _llm_mock
    except Exception:
        pass


@pytest.fixture
def mock_llm(monkeypatch):
    """Provide the LLM mock for explicit configuration in tests."""
    # Reset default side effects between tests
    _llm_mock.chat = AsyncMock(return_value="")
    _llm_mock.structured_call = AsyncMock(return_value=None)
    _llm_mock.json_chat = AsyncMock(return_value={})
    yield _llm_mock


# ===== 2. Redis Global Mock (module-level) =====

_redis_mock = AsyncMock()
_redis_mock.get_json = AsyncMock(return_value=None)
_redis_mock.set_json = AsyncMock()
_redis_mock.get = AsyncMock(return_value=None)
_redis_mock.set = AsyncMock()
_redis_mock.delete = AsyncMock()
_redis_mock.connect = AsyncMock()
_redis_mock.disconnect = AsyncMock()

import core.redis_client
core.redis_client.redis_client = _redis_mock

for _mod_name in [
    "api.main",
    "skills.poi_search",
    "skills.weather_query",
]:
    try:
        _mod = __import__(_mod_name, fromlist=["redis_client"])
        if hasattr(_mod, "redis_client"):
            _mod.redis_client = _redis_mock
    except Exception:
        pass


@pytest.fixture
def mock_redis(monkeypatch):
    """Provide the Redis mock for explicit configuration in tests."""
    _redis_mock.get_json = AsyncMock(return_value=None)
    _redis_mock.set_json = AsyncMock()
    _redis_mock.get = AsyncMock(return_value=None)
    _redis_mock.set = AsyncMock()
    _redis_mock.delete = AsyncMock()
    yield _redis_mock


# ===== 3. Database Session Mock =====

@pytest.fixture
def db_session():
    """Provide a mocked AsyncSession."""
    from unittest.mock import MagicMock
    session = AsyncMock()
    session.add = MagicMock()  # sync call
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    return session


# ===== 4. Standard Profile Fixtures =====

@pytest.fixture
def profile_beijing_3days():
    from schemas import UserProfile
    return UserProfile(
        destination="北京",
        travel_days=3,
        travel_dates="2026-05-01",
        travelers_count=2,
        travelers_type="情侣",
        budget_range=5000,
        food_preferences=["烤鸭", "涮肉"],
        interests=["历史", "文化", "美食"],
        pace="moderate",
    )


@pytest.fixture
def profile_chengdu_budget():
    from schemas import UserProfile
    return UserProfile(
        destination="成都",
        travel_days=4,
        travel_dates="2026-06-10",
        travelers_count=1,
        travelers_type="独自",
        budget_range=3000,
        food_preferences=["辣", "火锅", "串串"],
        interests=["美食", "拍照", "自然风光"],
        pace="relaxed",
    )


@pytest.fixture
def profile_minimal():
    from schemas import UserProfile
    return UserProfile(
        destination="上海",
        travel_days=2,
    )


# ===== 5. Standard POI Fixtures =====

@pytest.fixture
def poi_gugong():
    from schemas import ScoredPOI, Location
    return ScoredPOI(
        name="故宫博物院",
        category="attraction",
        score=0.95,
        location=Location(lat=39.916345, lng=116.397155),
        description="明清皇宫，世界文化遗产",
        tags=["历史", "文化", "皇家"],
        ticket_price=60,
        best_time="上午",
        area="东城区",
        recommended_hours="半天",
        indoor_outdoor="mixed",
    )


@pytest.fixture
def poi_changcheng():
    from schemas import ScoredPOI, Location
    return ScoredPOI(
        name="长城",
        category="attraction",
        score=0.95,
        location=Location(lat=40.359580, lng=116.019967),
        description="世界文化遗产",
        tags=["历史", "登山", "观景"],
        ticket_price=40,
        best_time="上午",
        area="延庆区",
        recommended_hours="半天",
        indoor_outdoor="outdoor",
    )


@pytest.fixture
def poi_sample_list(poi_gugong, poi_changcheng):
    from schemas import ScoredPOI, Location
    return [
        poi_gugong,
        poi_changcheng,
        ScoredPOI(
            name="全聚德",
            category="restaurant",
            score=0.8,
            location=Location(lat=39.899841, lng=116.398211),
            description="北京烤鸭老字号",
            tags=["烤鸭", "老字号", "必吃"],
            best_time="晚餐",
            area="前门",
            recommended_hours="1.5小时",
            indoor_outdoor="indoor",
        ),
        ScoredPOI(
            name="天坛",
            category="attraction",
            score=0.9,
            location=Location(lat=39.883455, lng=116.406588),
            description="明清祭天建筑",
            tags=["历史", "建筑", "文化"],
            ticket_price=34,
            best_time="上午",
            area="东城区",
            recommended_hours="2-3小时",
            indoor_outdoor="mixed",
        ),
    ]


# ===== 6. Standard Itinerary Fixtures =====

@pytest.fixture
def day_plan_beijing_day1(poi_gugong, poi_changcheng):
    from schemas import DayPlan, Activity
    return DayPlan(
        day_number=1,
        date="2026-05-01",
        theme="历史文化之旅",
        activities=[
            Activity(
                poi_name=poi_gugong.name,
                category="attraction",
                start_time="09:00",
                end_time="12:00",
                duration_min=180,
                ticket_price=60,
                recommendation_reason="世界文化遗产，必去",
                tags=["历史", "文化"],
            ),
            Activity(
                poi_name="午餐",
                category="restaurant",
                start_time="12:00",
                end_time="13:30",
                duration_min=90,
                meal_cost=80,
                recommendation_reason="品尝北京烤鸭",
                tags=["美食"],
            ),
            Activity(
                poi_name=poi_changcheng.name,
                category="attraction",
                start_time="14:00",
                end_time="17:00",
                duration_min=180,
                ticket_price=40,
                recommendation_reason="中华民族象征",
                tags=["历史", "登山"],
            ),
        ],
    )


@pytest.fixture
def itinerary_2days(day_plan_beijing_day1):
    from schemas import DayPlan, Activity
    day2 = DayPlan(
        day_number=2,
        date="2026-05-02",
        theme="皇家园林漫步",
        activities=[
            Activity(
                poi_name="颐和园",
                category="attraction",
                start_time="09:00",
                end_time="12:00",
                duration_min=180,
                ticket_price=30,
                recommendation_reason="皇家园林",
                tags=["园林", "湖景"],
            ),
        ],
    )
    return [day_plan_beijing_day1, day2]


# ===== 7. LLM Mock Helpers =====

@pytest.fixture
def mock_llm_for_intent(monkeypatch):
    """Configure mock LLM to return a specific IntentResult."""
    def _configure(intent, entities=None, missing_required=None, confidence=0.95):
        from schemas import IntentResult
        result = IntentResult(
            intent=intent,
            confidence=confidence,
            user_entities=entities or {},
            missing_required=missing_required or [],
        )
        mock = AsyncMock()
        mock.structured_call = AsyncMock(return_value=result)
        mock.chat = AsyncMock(return_value="")
        mock.json_chat = AsyncMock(return_value={})
        monkeypatch.setattr("core.llm_client.llm", mock)
        return mock
    return _configure


# ===== 8. FastAPI Test Client =====

@pytest.fixture
def client():
    """Provide a FastAPI TestClient with mocked lifespan."""
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    from api.routes import router as api_router
    from api.websocket import router as ws_router
    from fastapi.middleware.cors import CORSMiddleware

    test_app = FastAPI()
    test_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    test_app.include_router(api_router)
    test_app.include_router(ws_router)

    # Mock graph and checkpointer for tests
    test_app.state.graph = AsyncMock()
    test_app.state.checkpointer = AsyncMock()

    return TestClient(test_app)
