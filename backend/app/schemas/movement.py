"""Movement analysis schemas."""

from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from uuid import UUID
from pydantic import BaseModel, Field, model_validator


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

    session_type: str = Field(..., pattern="^(rehab_assessment|yoga|posture|gait|balance|flexibility)$")
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
    """Analysis request schema for movement assessment."""

    video_url: Optional[str] = None
    image_data: Optional[str] = None  # Base64 encoded (kept for testing / behind feature flag)
    exercise_type: str = Field(default="rehab_assessment")
    exercise_id: Optional[str] = "shoulder-abduction"
    user_id: Optional[UUID] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class MetricWithTag(BaseModel):
    """
    Individual metric with measured vs estimated tag.
    - 'measured': direct geometric computation (e.g., joint angle, ROM, symmetry).
    - 'estimated': inferred proxy or model prediction (e.g., quality score, stability score).
    """

    value: float
    tag: Literal["measured", "estimated"] = "measured"
    is_measured: bool = True
    description: Optional[str] = None
    unit: Optional[str] = None
    reference_mean: Optional[float] = None
    deviation: Optional[float] = None

    @model_validator(mode="before")
    @classmethod
    def sync_tags(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "tag" in data and "is_measured" not in data:
                data["is_measured"] = (data["tag"] == "measured")
            elif "is_measured" in data and "tag" not in data:
                data["tag"] = "measured" if data["is_measured"] else "estimated"
        return data


class AnalysisResponse(BaseModel):
    """Analysis response schema with measured/estimated tags and clinical explanations."""

    analysis_id: UUID
    exercise_type: str  # rehab_assessment, yoga, posture, gait, balance, flexibility
    overall_score: float  # 0-100 composite quality score
    score_breakdown: Dict[str, float]  # angle_accuracy, symmetry, stability, range_of_motion
    metrics: Dict[str, MetricWithTag]  # Individual metrics with measured/estimated tags
    feedback: List[str] = []  # Clinical and form feedback
    explanations: Dict[str, str] = {}  # Explanations for low scores (identifying joint/phase deviations)
    clinical_quality_score: Optional[float] = None  # Continuous score from KIMORE regression head
    movement_correct: Optional[bool] = None  # Binary classification from UI-PRMD classification head
    processing_time_ms: int
