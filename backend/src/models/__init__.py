"""SQLAlchemy ORM models aligned with PRD_AI全栈高并发改造."""

import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    String,
    Integer,
    DateTime,
    Text,
    JSON,
    Boolean,
    ForeignKey,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from core.database import Base


def _uuid_str() -> str:
    return str(uuid.uuid4())


class User(Base):
    """Registered and guest users."""

    __tablename__ = "users"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    email = Column(String(255), nullable=True, unique=True, index=True)
    phone = Column(String(20), nullable=True, unique=True, index=True)
    password_hash = Column(String(255), nullable=True)
    role = Column(
        String(20),
        nullable=False,
        default="guest",
        index=True,
    )  # guest / user / admin
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    profile = relationship("UserProfile", back_populates="user", uselist=False)
    conversations = relationship("Conversation", back_populates="user")
    itineraries = relationship("Itinerary", back_populates="user")
    planning_jobs = relationship("PlanningJob", back_populates="user")


class UserProfile(Base):
    """User preference and personalization profile."""

    __tablename__ = "user_profiles"

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    personal = Column(JSON, default=dict, nullable=False)
    preferences = Column(JSON, default=dict, nullable=False)
    frequent_destinations = Column(JSON, default=list, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    user = relationship("User", back_populates="profile")


class Conversation(Base):
    """A multi-turn conversation/session."""

    __tablename__ = "conversations"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title = Column(String(255), nullable=True)
    status = Column(
        String(20),
        nullable=False,
        default="active",
        index=True,
    )  # active / archived / deleted
    state_snapshot = Column(JSON, default=dict, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )
    archived_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="conversations")
    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
    )
    planning_jobs = relationship("PlanningJob", back_populates="conversation")
    itineraries = relationship("Itinerary", back_populates="conversation")

    __table_args__ = (Index("ix_conversations_user_updated", "user_id", "updated_at"),)


class Message(Base):
    """A single message within a conversation."""

    __tablename__ = "messages"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role = Column(String(20), nullable=False)  # user / assistant / system
    content = Column(Text, nullable=False)
    token_count = Column(Integer, default=0, nullable=False)
    metadata_ = Column("metadata", JSON, default=dict, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    conversation = relationship("Conversation", back_populates="messages")

    __table_args__ = (Index("ix_messages_conversation_created", "conversation_id", "created_at"),)


class Itinerary(Base):
    """Generated travel itinerary."""

    __tablename__ = "itineraries"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    job_id = Column(
        String(36),
        ForeignKey("planning_jobs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    destination = Column(String(100), nullable=False, index=True)
    days = Column(Integer, nullable=False)
    content = Column(JSON, default=dict, nullable=False)
    proposal_text = Column(Text, nullable=True)
    is_favorite = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    user = relationship("User", back_populates="itineraries")
    conversation = relationship("Conversation", back_populates="itineraries")
    job = relationship("PlanningJob", back_populates="itineraries")

    __table_args__ = (Index("ix_itineraries_user_created", "user_id", "created_at"),)


# Import data-layer and log models at the end to avoid circular imports and
# register them with Base metadata. These tables already exist in migrations
# but were missing from ORM, which caused alembic autogenerate to drop them.
from models.attraction import Attraction  # noqa: E402, F401
from models.city_info import CityInfo  # noqa: E402, F401
from models.data_audit_log import DataAuditLog  # noqa: E402, F401
from models.dead_letter_archive import DeadLetterArchive  # noqa: E402, F401
from models.hotel import Hotel  # noqa: E402, F401
from models.knowledge_tip import KnowledgeTip  # noqa: E402, F401
from models.planning_job import PlanningJob, PlanningJobEvent  # noqa: E402, F401
from models.planning_log import PlanningLog  # noqa: E402, F401
from models.restaurant import Restaurant  # noqa: E402, F401
from models.spot_distance import SpotDistanceMulti  # noqa: E402, F401
from models.transport_hub import TransportHub  # noqa: E402, F401
from models.user_modification_log import UserModificationLog  # noqa: E402, F401
from models.user_profile_vector import UserProfileVector  # noqa: E402, F401
from models.user_trip_history import UserTripHistory  # noqa: E402, F401

__all__ = [
    "User",
    "UserProfile",
    "UserProfileVector",
    "Conversation",
    "Message",
    "Itinerary",
    "PlanningJob",
    "PlanningJobEvent",
    "DeadLetterArchive",
    "Attraction",
    "Restaurant",
    "Hotel",
    "KnowledgeTip",
    "CityInfo",
    "TransportHub",
    "DataAuditLog",
    "SpotDistanceMulti",
    "UserTripHistory",
    "PlanningLog",
    "UserModificationLog",
]
