"""Health insights API endpoints."""

from fastapi import APIRouter, Depends

from app.api.deps import get_current_user_dependency
from app.schemas.insight import DashboardData, InsightRead, TrendDataPoint

router = APIRouter()


@router.get("/dashboard", response_model=DashboardData)
async def get_dashboard(
    current_user=Depends(get_current_user_dependency),
):
    """
    Get dashboard data for the current user.
    
    Returns health score, progress, streaks, and recent activity.
    """
    # Placeholder - implement with actual data
    return DashboardData(
        health_score=85.5,
        weekly_progress=7.5,
        current_streak=5,
        total_sessions=42,
        recent_activities=[],
        upcoming_challenges=[],
    )


@router.get("/list", response_model=list[InsightRead])
async def list_insights(
    current_user=Depends(get_current_user_dependency),
    limit: int = 50,
    seen: bool = None,
):
    """
    List health insights for the current user.
    
    - **limit**: Maximum number of insights
    - **seen**: Filter by seen status (true/false/all)
    """
    # Placeholder
    return []


@router.get("/trends", response_model=list[TrendDataPoint])
async def get_trends(
    current_user=Depends(get_current_user_dependency),
    metric_type: str = "all",
    days: int = 30,
):
    """
    Get trend data for the current user.
    
    - **metric_type**: Type of metric to retrieve
    - **days**: Number of days to include
    """
    # Placeholder
    return []
