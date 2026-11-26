"""User-related database models."""

from datetime import datetime
from sqlalchemy import Column, String, Boolean, Date, DateTime, ForeignKey, JSON, Uuid
from sqlalchemy.orm import relationship

from app.models.base import BaseModel


class User(BaseModel):
    """User account model."""
    
    __tablename__ = "users"

    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    date_of_birth = Column(Date)
    gender = Column(String(20))
    height_cm = Column(String(20))
    weight_kg = Column(String(20))
    activity_level = Column(String(20))
    is_active = Column(Boolean, default=True, nullable=False)
    is_email_verified = Column(Boolean, default=False, nullable=False)
    last_login_at = Column(DateTime)
    preferences = Column(String, default="{}")

    # Relationships
    profile = relationship("Profile", back_populates="user", uselist=False)
    consents = relationship("Consent", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("MovementSession", back_populates="user", cascade="all, delete-orphan")
    subscriptions = relationship("Subscription", back_populates="user", cascade="all, delete-orphan")


class Profile(BaseModel):
    """User profile model with detailed health information."""
    
    __tablename__ = "profiles"

    user_id = Column(Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    injury_history = Column(JSON, default=list)
    medical_conditions = Column(JSON, default=list)
    fitness_goals = Column(String, default="{}")
    posture_goals = Column(String, default="{}")

    user = relationship("User", back_populates="profile")


class Consent(BaseModel):
    """User consent model for data processing."""
    
    __tablename__ = "consents"

    user_id = Column(Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    consent_type = Column(String(50), nullable=False)
    status = Column(String(20), nullable=False)
    granted_at = Column(DateTime)
    revoked_at = Column(DateTime)
    version = Column(String(20))
    record_metadata = Column("metadata", String, default="{}")

    user = relationship("User", back_populates="consents")


class Subscription(BaseModel):
    """User subscription model."""
    
    __tablename__ = "subscriptions"

    user_id = Column(Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    provider_subscription_id = Column(String(255), unique=True)
    plan_id = Column(String(50), nullable=False)
    status = Column(String(20), nullable=False)
    current_period_start = Column(DateTime)
    current_period_end = Column(DateTime)
    cancel_at_period_end = Column(Boolean, default=False, nullable=False)

    user = relationship("User", back_populates="subscriptions")
