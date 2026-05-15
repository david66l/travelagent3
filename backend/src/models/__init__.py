import uuid
from datetime import datetime

from sqlalchemy import Column, String, Integer, DateTime, Text, JSON

from core.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(String(100), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(100), nullable=False, unique=True)
    email = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Session(Base):
    __tablename__ = "sessions"

    id = Column(String(100), primary_key=True)
    user_id = Column(String(100), nullable=False)
    destination = Column(String(100), nullable=True)
    travel_days = Column(Integer, nullable=True)
    status = Column(String(50), default="active")  # active / completed / abandoned
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String(100), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(100), nullable=False)
    user_message = Column(Text, nullable=False)
    assistant_response = Column(Text, nullable=True)
    intent = Column(String(50), nullable=True)
    tools_used = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)


class Itinerary(Base):
    __tablename__ = "itineraries"

    id = Column(String(100), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(100), nullable=False)
    user_id = Column(String(100), nullable=False)
    destination = Column(String(100), nullable=False)
    travel_days = Column(Integer, nullable=False)
    daily_plans = Column(JSON, default=list)
    preference_snapshot = Column(JSON, default=dict)
    budget_snapshot = Column(JSON, default=dict)
    status = Column(String(50), default="draft")  # draft / confirmed / completed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PreferenceChange(Base):
    __tablename__ = "preference_changes"

    id = Column(String(100), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(100), nullable=False)
    user_id = Column(String(100), nullable=False)
    field = Column(String(100), nullable=False)
    old_value = Column(Text, nullable=True)
    new_value = Column(Text, nullable=False)
    source_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# Register new planning job models with Base metadata
from models.planning_job import PlanningJob, PlanningJobEvent  # noqa: E402, F401
