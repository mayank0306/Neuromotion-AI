"""Movement analysis schemas."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field


class PoseLandmarkSchema(BaseModel):
    """Pose landmark schema."""
    
    x: float
    y: float
    z: float = 0.0
    visibility: float = 0.0
    presence: float = 0.0


class PoseDataSchema(BaseModel):
    """Pose data for a single frame."""
    
    frame_index: int
    timestamp: datetime
    landmarks: List[PoseLandmarkSchema]
    score: Optional[float] = None


class MovementSessionBase(BaseModel):
    """Base movement session schema."""

    session_type: str = Field(..., pattern="^(yoga|posture|gait|balance|flexibility|rehab_assessment)$")
    exercise_id: Optional[str] = None


class MovementSessionCreate(MovementSessionBase):
    """Session creation schema."""
    
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class MovementSessionRead(MovementSessionBase):
    """Session read schema."""
    
    id: UUID
    user_id: UUID
    duration_seconds: Optional[int] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    score: Optional[float] = None
    created_at: datetime
    
    model_config = {"from_attributes": True}


class ExerciseSchema(BaseModel):
    """Exercise library schema."""
    
    id: str
    name: str
    description: Optional[str] = None
    category: str
    difficulty: str
    duration_seconds: Optional[int] = None
    target_muscles: List[str] = []
    instructions: Optional[Dict[str, Any]] = None
    video_url: Optional[str] = None
    thumbnail_url: Optional[str] = None


class AnalysisRequest(BaseModel):
    """Analysis request schema."""
    
    video_url: Optional[str] = None
    image_data: Optional[str] = None  # Base64 encoded
    exercise_type: str
    user_id: UUID
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class MetricWithTag(BaseModel):
    """Individual metric with measured/estimated tag."""

    value: float
    is_measured: bool  # True for direct geometric computation, False for inferred proxy
    description: str = None


class AnalysisResponse(BaseModel):
    """Analysis response schema."""

    analysis_id: UUID
    exercise_type: str  # yoga, posture, gait, balance, flexibility, rehab_assessment
    overall_score: float  # 0-100 quality score
    score_breakdown: Dict[str, float]  # Individual component scores (angle_accuracy, symmetry, etc.)
    metrics: Dict[str, MetricWithTag]  # Individual metrics with measured/estimated tags
    feedback: List[str] = []  # Human-readable feedback
    explanations: Dict[str, str] = {}  # Explanations for low scores (joint/phase causing issues)
    processing_time_ms: int
