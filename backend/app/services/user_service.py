"""User service for profile management."""

from typing import Optional
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from app.models.user import User
from app.schemas.user import UserUpdate, UserRead

logger = structlog.get_logger(__name__)


class UserService:
    """Handles user profile operations."""
    
    def __init__(self, db: AsyncSession, redis_client=None):
        self.db = db
        self.redis = redis_client

    async def get_user(self, user_id: UUID) -> Optional[UserRead]:
        """Get user by ID."""
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            return None
        
        return UserRead.model_validate(user)

    async def update_user(self, user_id: UUID, data: UserUpdate) -> Optional[UserRead]:
        """Update user profile."""
        stmt = (
            update(User)
            .where(User.id == user_id)
            .values(**data.model_dump(exclude_unset=True))
        )
        
        await self.db.execute(stmt)
        await self.db.commit()
        
        # Fetch updated user
        return await self.get_user(user_id)
