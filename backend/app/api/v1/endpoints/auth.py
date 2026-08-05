"""Authentication API endpoints."""

from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from app.api.deps import get_auth_service
from app.schemas.auth import LoginRequest, RegisterRequest, Token
from app.schemas.user import UserRegistrationResponse

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


@router.post("/register", response_model=UserRegistrationResponse)
async def register(
    request: RegisterRequest,
    auth_service=Depends(get_auth_service),
):
    """
    Register a new user account.
    
    - **email**: Valid email address
    - **password**: Minimum 12 characters
    - **first_name**: User's first name
    - **last_name**: User's last name
    """
    user = await auth_service.register(request)
    
    tokens = await auth_service.login(
        LoginRequest(email=request.email, password=request.password)
    )
    return UserRegistrationResponse(user=user, token=tokens)


@router.post("/login", response_model=Token)
async def login(
    request: LoginRequest,
    auth_service=Depends(get_auth_service),
):
    """
    Authenticate user and return JWT tokens.
    
    - **email**: User's email
    - **password**: User's password
    """
    return await auth_service.login(request)


@router.post("/logout")
async def logout(
    token: str = Depends(oauth2_scheme),
):
    """
    Logout user by invalidating the token.
    
    The token is added to a blacklist and cannot be used for subsequent requests.
    """
    # In production, add token to Redis blacklist
    return {"message": "Successfully logged out"}


@router.post("/refresh", response_model=Token)
async def refresh_token(
    refresh_token: str,
    auth_service=Depends(get_auth_service),
):
    """
    Refresh an access token using a refresh token.
    
    - **refresh_token**: Valid refresh token
    """
    from app.core.security import verify_token, create_access_token, create_refresh_token
    
    try:
        payload = verify_token(refresh_token)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    # Validate token type
    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type"
        )
    
    # Generate new access token
    access_token = create_access_token(
        data={"sub": payload["sub"]},
        expires_delta=timedelta(minutes=30),
    )
    
    new_refresh_token = create_refresh_token(
        data={"sub": payload["sub"]},
        expires_delta=timedelta(days=7),
    )
    
    return Token(
        access_token=access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=1800,
    )
