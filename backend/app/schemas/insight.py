"""Insight and analytics schemas."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID
from pydantic import BaseModel


class InsightRead(BaseModel):
    """Insight read schema."""
    
    id: UUID
    user_id: UUID
    insight_type: str
    severity: str
    title: str
    description: str
    action_items: List[str]
    related_metrics: List[Dict[str, Any]]
    seen: bool
    acknowledged: bool
    created_at: datetime
    
    model_config = {"from_attributes": True}


class DashboardData(BaseModel):
    """Dashboard data schema."""
    
    health_score: float
    weekly_progress: float
    current_streak: int
    total_sessions: int
    recent_activities: List[Dict[str, Any]]
    upcoming_challenges: List[Dict[str, Any]]


class TrendDataPoint(BaseModel):
    """Single trend data point."""
    
    date: datetime
    value: float
    metric_type: str
