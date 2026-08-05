"""Authentication service for user login and token management."""

from datetime import datetime, timedelta
from typing import Optional

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import hash_password, verify_password, create_access_token, create_refresh_token
from app.exceptions import AuthenticationError, ConflictError
from app.models.user import User
from app.schemas.auth import LoginRequest, RegisterRequest, Token

logger = structlog.get_logger(__name__)


class AuthService:
    """Handles authentication operations."""
    
    def __init__(self, db: AsyncSession, redis_client=None):
        self.db = db
        self.redis = redis_client

    async def login(self, request: LoginRequest) -> Token:
        """
        Authenticate user and return tokens.
        
        Args:
            request: Login request with email and password.
            
        Returns:
            Token pair (access and refresh).
            
        Raises:
            AuthenticationError: If credentials are invalid.
        """
        # Find user by email
        result = await self.db.execute(
            select(User).where(User.email == request.email)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            logger.warning("Login attempt with invalid email", email=request.email)
            raise AuthenticationError("Invalid credentials")
        
        # Verify password
        if not verify_password(request.password, user.password_hash):
            logger.warning("Login attempt with invalid password", user_id=str(user.id))
            raise AuthenticationError("Invalid credentials")
        
        # Update last login
        user.last_login_at = datetime.utcnow()
        await self.db.commit()
        
        # Generate tokens
        access_token = create_access_token(
            data={"sub": str(user.id), "email": user.email},
            expires_delta=timedelta(minutes=30),
        )
        refresh_token = create_refresh_token(
            data={"sub": str(user.id)},
            expires_delta=timedelta(days=7),
        )
        
        logger.info("User logged in successfully", user_id=str(user.id))
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=1800,  # 30 minutes
        )

    async def register(self, request: RegisterRequest):
        """
        Register new user.
        
        Args:
            request: Registration data.
            
        Returns:
            Created user instance.
            
        Raises:
            ConflictError: If user already exists.
        """
        # Check if user exists
        result = await self.db.execute(
            select(User).where(User.email == request.email)
        )
        if result.scalar_one_or_none():
            raise ConflictError("User with this email already exists")
        
        # Create user
        user = User(
            email=request.email,
            password_hash=hash_password(request.password),
            first_name=request.first_name,
            last_name=request.last_name,
            date_of_birth=request.date_of_birth,
            gender=request.gender,
        )
        
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        
        logger.info("User registered successfully", user_id=str(user.id))
        
        return user

    async def validate_token(self, token: str) -> Optional[str]:
        """
        Validate access token and return user ID.
        
        Args:
            token: JWT access token.
            
        Returns:
            User ID if valid, None otherwise.
        """
        from app.core.security import verify_token
        
        try:
            payload = verify_token(token)
            return payload.get("sub")
        except Exception:
            return None
