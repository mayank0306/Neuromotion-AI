"""API dependencies for FastAPI."""

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from app.core.database import get_db, get_cache
from app.core.config import settings
from app.core.security import verify_token
from app.exceptions import AuthenticationError
from app.models.user import User

from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis
from app.services.auth_service import AuthService
from app.services.user_service import UserService
from app.services.movement_service import MovementService
from app.ai.pose_estimator import get_pose_estimator


# Database dependencies
DatabaseSession = Annotated[AsyncSession, Depends(get_db)]
def get_auth_service(
    db: DatabaseSession,
) -> AuthService:
    """Dependency for AuthService."""
    return AuthService(db)


def get_user_service(
    db: DatabaseSession,
) -> UserService:
    """Dependency for UserService."""
    return UserService(db)


def get_movement_service(
    db: DatabaseSession,
) -> MovementService:
    """Dependency for MovementService."""
    return MovementService(db)


def get_ai_service():
    """Dependency for AI Service."""
    return get_pose_estimator()


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


async def get_current_user_dependency(
    db: DatabaseSession,
    token: str = Depends(oauth2_scheme),
) -> User:
    """
    Get currently authenticated user.
    
    Extracts and validates JWT token from request,
    returns the authenticated user.
    """
    try:
        payload = verify_token(token)
        if payload.get("type") != "access":
            raise AuthenticationError("Invalid token type")
        user_id = payload.get("sub")
        if not user_id:
            raise AuthenticationError("Invalid token payload")
        user = await db.get(User, user_id)
    except AuthenticationError:
        raise
    except Exception as exc:
        raise AuthenticationError("Invalid or expired token") from exc

    if user is None or not user.is_active:
        raise AuthenticationError("User account is unavailable")
    return user


# Token verification utility
def verify_access_token(token: str) -> dict:
    """
    Verify access token and return payload.
    
    Args:
        token: JWT access token.
        
    Returns:
        Token payload data.
        
    Raises:
        AuthenticationError: If token is invalid or expired.
    """
    try:
        payload = verify_token(token)
        return payload
    except Exception:
        raise AuthenticationError("Invalid or expired token")
