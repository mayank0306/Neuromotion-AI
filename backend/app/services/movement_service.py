"""Movement service for session and pose management."""

from typing import Optional, List
from uuid import UUID
from datetime import datetime

from sqlalchemy import select, insert
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from app.models.movement import MovementSession, Pose, Analysis
from app.schemas.movement import MovementSessionCreate, MovementSessionRead

logger = structlog.get_logger(__name__)


class MovementService:
    """Handles movement session operations."""
    
    def __init__(self, db: AsyncSession, redis_client=None):
        self.db = db
        self.redis = redis_client

    async def create_session(
        self, user_id: UUID, data: MovementSessionCreate
    ) -> MovementSessionRead:
        """Create a new movement session."""
        session = MovementSession(
            user_id=user_id,
            session_type=data.session_type,
            exercise_id=data.exercise_id,
            start_time=datetime.utcnow(),
            session_metadata=data.metadata or {},
        )
        
        self.db.add(session)
        await self.db.commit()
        await self.db.refresh(session)
        
        logger.info("Session created", user_id=str(user_id), session_id=str(session.id))
        
        return MovementSessionRead.model_validate(session)

    async def get_session(self, session_id: UUID) -> Optional[MovementSessionRead]:
        """Get session by ID."""
        result = await self.db.execute(
            select(MovementSession).where(MovementSession.id == session_id)
        )
        session = result.scalar_one_or_none()
        
        if not session:
            return None
        
        return MovementSessionRead.model_validate(session)

    async def list_sessions(
        self, user_id: UUID, limit: int = 50, offset: int = 0
    ) -> List[MovementSessionRead]:
        """List sessions for a user."""
        result = await self.db.execute(
            select(MovementSession)
            .where(MovementSession.user_id == user_id)
            .order_by(MovementSession.start_time.desc())
            .offset(offset)
            .limit(limit)
        )
        
        sessions = result.scalars().all()
        return [MovementSessionRead.model_validate(s) for s in sessions]

    async def add_poses(
        self, session_id: UUID, poses: List[Pose]
    ) -> None:
        """Add poses to a session."""
        for pose in poses:
            pose.session_id = session_id
            self.db.add(pose)
        
        await self.db.commit()

    async def complete_session(
        self, session_id: UUID, score: float, duration: int
    ) -> Optional[MovementSessionRead]:
        """Mark session as complete with final score."""
        from sqlalchemy import update
        
        stmt = (
            update(MovementSession)
            .where(MovementSession.id == session_id)
            .values(
                score=score,
                duration_seconds=duration,
                end_time=datetime.utcnow(),
            )
        )
        
        await self.db.execute(stmt)
        await self.db.commit()
        
        return await self.get_session(session_id)
