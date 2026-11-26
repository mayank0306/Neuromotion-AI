"""Movement-related database models."""

from datetime import datetime
from sqlalchemy import Boolean, Column, String, Integer, Float, Text, DateTime, ForeignKey, JSON, Uuid
from sqlalchemy.orm import relationship

from app.models.base import BaseModel


class MovementSession(BaseModel):
    """Movement session tracking."""
    
    __tablename__ = "sessions"

    user_id = Column(Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    session_type = Column(String(50), nullable=False, index=True)  # yoga, posture, gait, balance
    exercise_id = Column(String(100))
    duration_seconds = Column(Integer)
    start_time = Column(DateTime, index=True)
    end_time = Column(DateTime)
    score = Column(Float)
    session_metadata = Column("metadata", JSON, default=dict)

    user = relationship("User", back_populates="sessions")
    poses = relationship("Pose", back_populates="session", cascade="all, delete-orphan")
    analyses = relationship("Analysis", back_populates="session", cascade="all, delete-orphan")


class Pose(BaseModel):
    """Pose detection data for a frame."""
    
    __tablename__ = "poses"

    session_id = Column(Uuid(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    frame_index = Column(Integer, nullable=False)
    timestamp = Column(DateTime, index=True)
    landmarks = Column(JSON)  # Array of landmark objects
    score = Column(Float)
    world_landmarks = Column(JSON)
    visibility = Column(JSON)

    session = relationship("MovementSession", back_populates="poses")


class Exercise(BaseModel):
    """Exercise library entry."""
    
    __tablename__ = "exercises"

    id = Column(String(100), primary_key=True)  # Natural key
    name = Column(String(255), nullable=False)
    description = Column(Text)
    category = Column(String(50), index=True)  # yoga, strength, flexibility
    difficulty = Column(String(20))  # beginner, intermediate, advanced
    duration_seconds = Column(Integer)
    target_muscles = Column(JSON, default=list)
    instructions = Column(JSON)
    video_url = Column(String(500))
    thumbnail_url = Column(String(500))
    is_active = Column(Boolean, default=True, nullable=False)


class Analysis(BaseModel):
    """Analysis results for a session."""

    __tablename__ = "analyses"

    session_id = Column(Uuid(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    analysis_type = Column(String(50), nullable=False)  # yoga, posture, gait, balance, flexibility, rehab_assessment
    score = Column(Float)  # Overall quality score (0-100)
    score_breakdown = Column(JSON)  # Individual component scores
    metrics = Column(JSON)  # Detailed metrics with measured/estimated tags
    feedback = Column(JSON)  # Detailed feedback array
    explanations = Column(JSON)  # Explanations for low scores (joint/phase causing issues)
    completed_at = Column(DateTime)

    session = relationship("MovementSession", back_populates="analyses")
