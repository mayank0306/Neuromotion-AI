"""User management API endpoints."""

from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_current_user_dependency, get_user_service
from app.schemas.user import UserRead, UserUpdate

router = APIRouter()


@router.get("/me", response_model=UserRead)
async def get_current_user(
    current_user=Depends(get_current_user_dependency),
):
    """
    Get current user profile.
    
    Returns the authenticated user's profile information.
    """
    return current_user


@router.patch("/me", response_model=UserRead)
async def update_current_user(
    data: UserUpdate,
    current_user=Depends(get_current_user_dependency),
    user_service=Depends(get_user_service),
):
    """
    Update current user profile.
    
    Fields are partial - only provided values will be updated.
    """
    return await user_service.update_user(current_user.id, data)


@router.get("/{user_id}", response_model=UserRead)
async def get_user(
    user_id: UUID,
    current_user=Depends(get_current_user_dependency),
    user_service=Depends(get_user_service),
):
    """
    Get user by ID.
    
    Users can only access their own profiles unless they are an admin.
    """
    if str(user_id) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot access another user's profile"
        )
    
    user = await user_service.get_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return user
