"""User schemas."""

from datetime import datetime
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel, EmailStr, Field

from app.schemas.auth import Token


class UserBase(BaseModel):
    """Base user schema with common fields."""
    
    email: EmailStr
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)


class UserCreate(UserBase):
    """User creation schema."""
    
    password: str = Field(..., min_length=12)


class UserRead(UserBase):
    """User read schema."""
    
    id: UUID
    is_active: bool
    is_email_verified: bool
    last_login_at: Optional[datetime] = None
    created_at: datetime
    
    model_config = {"from_attributes": True}


class UserUpdate(BaseModel):
    """User update schema."""
    
    first_name: Optional[str] = Field(None, min_length=1, max_length=100)
    last_name: Optional[str] = Field(None, min_length=1, max_length=100)
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    height_cm: Optional[float] = Field(None, ge=50, le=250)
    weight_kg: Optional[float] = Field(None, ge=20, le=300)
    activity_level: Optional[str] = None


class UserReadWithProfile(UserRead):
    """User with profile data."""
    
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    activity_level: Optional[str] = None
    preferences: dict = {}


class UserRegistrationResponse(BaseModel):
    """User registration response with token."""
    
    user: UserRead
    token: Token
