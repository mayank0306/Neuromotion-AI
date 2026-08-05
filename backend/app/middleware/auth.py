"""
Authentication middleware.

Validates JWT tokens and sets user context for requests.
"""

from typing import Optional
import structlog

from fastapi import Request
from starlette.types import ASGIApp, Receive, Scope, Send

from app.core.config import settings
from app.core.security import verify_token

logger = structlog.get_logger(__name__)


class AuthMiddleware:
    """
    Middleware for JWT authentication.
    
    Extracts and validates JWT tokens from Authorization header or cookies.
    Sets the user ID in the request scope for downstream handlers.
    """
    
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive, send)
        
        # Extract token from Authorization header
        auth_header = scope.get("headers", [])
        token = self._extract_token(auth_header)
        
        if token:
            try:
                payload = verify_token(token)
                scope["user"] = payload
            except Exception as e:
                logger.warning("Invalid token provided", error=str(e))
                scope["user"] = None
        else:
            scope["user"] = None
        
        await self.app(scope, receive, send)

    def _extract_token(self, headers: list) -> Optional[str]:
        """Extract JWT token from headers."""
        for header_name, header_value in headers:
            if header_name == b"authorization":
                value = header_value.decode("utf-8")
                if value.startswith("Bearer "):
                    return value[7:]
        return None
