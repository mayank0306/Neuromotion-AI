"""Authentication schemas."""

from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class Token(BaseModel):
    """JWT token response."""
    
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class TokenPayload(BaseModel):
    """JWT token payload."""
    
    sub: str  # User ID
    exp: int  # Expiration timestamp
    iat: int  # Issued at timestamp
    scopes: list[str] = []
    token_type: str


class LoginRequest(BaseModel):
    """User login request."""
    
    email: EmailStr
    password: str = Field(..., min_length=12)


class RegisterRequest(BaseModel):
    """User registration request."""
    
    email: EmailStr
    password: str = Field(..., min_length=12, max_length=128)
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None


class RefreshTokenRequest(BaseModel):
    """Refresh token request."""
    
    refresh_token: str
