"""
Logging middleware for request tracking.

Logs incoming requests and outgoing responses with timing information.
"""

import time
import structlog
from fastapi import Request

logger = structlog.get_logger(__name__)


class RequestLoggingMiddleware:
    """
    Middleware that logs all incoming requests and responses.
    
    Adds request timing, method, path, and status code to logs.
    """
    
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Generate request ID
        import uuid
        request_id = str(uuid.uuid4())
        
        # Add to scope for logging
        scope["state"]["request_id"] = request_id
        
        # Get request details
        method = scope.get("method", "UNKNOWN")
        path = scope.get("path", "/")
        
        start_time = time.perf_counter()
        
        logger.info(
            "Request started",
            request_id=request_id,
            method=method,
            path=path,
        )
        
        # Capture response
        status_code = [None]
        
        async def _logging_send(message):
            if message["type"] == "http.response.start":
                status_code[0] = message.get("status")
            await send(message)
        
        try:
            await self.app(scope, receive, _logging_send)
            
            duration = time.perf_counter() - start_time
            
            logger.info(
                "Request completed",
                request_id=request_id,
                method=method,
                path=path,
                status_code=status_code[0],
                duration_ms=round(duration * 1000, 2),
            )
        except Exception as e:
            duration = time.perf_counter() - start_time
            logger.error(
                "Request failed",
                request_id=request_id,
                method=method,
                path=path,
                error=str(e),
                duration_ms=round(duration * 1000, 2),
            )
            raise
