"""Movement analysis API endpoints."""

from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import Field

from app.api.deps import get_current_user_dependency, get_movement_service
from app.schemas.movement import (
    MovementSessionCreate,
    MovementSessionRead,
    ExerciseSchema,
)

router = APIRouter()


@router.post("/sessions", response_model=MovementSessionRead, status_code=201)
async def create_session(
    request: MovementSessionCreate,
    current_user=Depends(get_current_user_dependency),
    movement_service=Depends(get_movement_service),
):
    """
    Create a new movement session.
    
    - **session_type**: Type of session (yoga, posture, gait, balance)
    - **exercise_id**: Optional exercise identifier
    - **metadata**: Optional additional data for the session
    """
    return await movement_service.create_session(current_user.id, request)


@router.get("/sessions", response_model=List[MovementSessionRead])
async def list_sessions(
    limit: int = 50,
    offset: int = 0,
    current_user=Depends(get_current_user_dependency),
    movement_service=Depends(get_movement_service),
):
    """
    List all movement sessions for the current user.
    
    - **limit**: Maximum number of sessions (default: 50)
    - **offset**: Number of sessions to skip (default: 0)
    """
    return await movement_service.list_sessions(
        current_user.id, limit=limit, offset=offset
    )


@router.get("/sessions/{session_id}", response_model=MovementSessionRead)
async def get_session(
    session_id: UUID,
    current_user=Depends(get_current_user_dependency),
    movement_service=Depends(get_movement_service),
):
    """Get a specific session by ID."""
    session = await movement_service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    if str(session.user_id) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot access another user's session"
        )
    
    return session


@router.get("/exercises", response_model=List[ExerciseSchema])
async def list_exercises(
    category: str = None,
    difficulty: str = None,
    current_user=Depends(get_current_user_dependency),
):
    """
    List all available exercises.
    
    - **category**: Filter by category (yoga, strength, flexibility)
    - **difficulty**: Filter by difficulty (beginner, intermediate, advanced)
    """
    # Placeholder - implement with actual exercise library
    from app.data.exercises import EXERCISE_LIBRARY
    exercises = EXERCISE_LIBRARY
    
    if category:
        exercises = [e for e in exercises if e["category"] == category]
    if difficulty:
        exercises = [e for e in exercises if e["difficulty"] == difficulty]
    
    return exercises
