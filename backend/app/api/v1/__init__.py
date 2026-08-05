"""API v1 routes."""

from fastapi import APIRouter

from app.api.v1.endpoints import auth, users, movement, analysis, insights

api_router = APIRouter()

# Include all route modules
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(movement.router, prefix="/movement", tags=["movement"])
api_router.include_router(analysis.router, prefix="/analysis", tags=["analysis"])
api_router.include_router(insights.router, prefix="/insights", tags=["insights"])
