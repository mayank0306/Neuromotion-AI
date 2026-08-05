"""
Security utilities for authentication and authorization.

Provides:
- Password hashing and verification
- JWT token creation and validation
- Token generation utilities
"""

from datetime import datetime, timedelta
from typing import Optional, Union

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import settings

# PBKDF2-SHA256 is broadly supported by passlib and uses a per-password salt.
# The high work factor helps resist offline guessing attacks without coupling
# the application to a platform-specific bcrypt binary.
pwd_context = CryptContext(
    schemes=["pbkdf2_sha256"],
    deprecated="auto",
    pbkdf2_sha256__rounds=310_000,
)


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.

    Args:
        password: Plain text password to hash.

    Returns:
        Hashed password string.
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.

    Args:
        plain_password: Plain text password to verify.
        hashed_password: Hashed password to compare against.

    Returns:
        True if password matches, False otherwise.
    """
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a JWT access token.

    Args:
        data: Payload data to encode in the token.
        expires_delta: Token expiration time (defaults to settings).

    Returns:
        Encoded JWT access token.
    """
    from copy import copy

    to_encode = copy(data)
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access",
    })

    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )

    return encoded_jwt


def create_refresh_token(
    data: dict,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a JWT refresh token.

    Args:
        data: Payload data to encode in the token.
        expires_delta: Token expiration time (defaults to settings).

    Returns:
        Encoded JWT refresh token.
    """
    from copy import copy

    to_encode = copy(data)
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS
        )

    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh",
    })

    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )

    return encoded_jwt


def verify_token(token: str) -> dict:
    """
    Verify a JWT token and return its payload.

    Args:
        token: JWT token to verify.

    Returns:
        Decoded token payload.

    Raises:
        jwt.InvalidTokenError: If token is invalid.
        jwt.ExpiredSignatureError: If token is expired.
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        return payload
    except JWTError:
        raise jwt.InvalidTokenError("Invalid token")


def generate_token_data(user_id: str, scopes: list = None) -> dict:
    """
    Generate standard token payload data.

    Args:
        user_id: User ID to include in token.
        scopes: List of permission scopes.

    Returns:
        Dictionary with standard token claims.
    """
    return {
        "sub": user_id,
        "scopes": scopes or ["read"],
        "token_type": "bearer",
    }
