"""
NeuroMotion Backend Application Main Module.

This module initializes the FastAPI application with all middleware,
routes, and configuration.
"""

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog

from app.core.config import settings
from app.core.database import init_db
from app.core.logging import configure_logging
from app.api.v1 import api_router
from app.exceptions import AppException


# ┌──────────────────────────────────────────────────────────────┐
# │ Configure logging                                            │
# └──────────────────────────────────────────────────────────────┘
configure_logging()
logger = structlog.get_logger(__name__)


# ┌──────────────────────────────────────────────────────────────┐
# │ Application lifespan (startup/shutdown)                     │
# └──────────────────────────────────────────────────────────────┘
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle events.

    Startup:
    - Initialize database connections
    - Load AI models
    - Warm up caches
    - Log startup information

    Shutdown:
    - Close database connections
    - Release AI model resources
    - Log shutdown information
    """
    logger.info("Starting NeuroMotion backend...")

    # Initialize database
    await init_db()
    logger.info("Database initialized")

    # Load AI models (placeholder)
    # await load_ai_models()
    # logger.info("AI models loaded")

    # Health check
    logger.info("NeuroMotion backend is ready", version=settings.APP_VERSION)

    yield

    logger.info("Shutting down NeuroMotion backend...")
    # Cleanup resources here
    logger.info("NeuroMotion backend shut down successfully")


# ┌──────────────────────────────────────────────────────────────┐
# │ FastAPI Application                                          │
# └──────────────────────────────────────────────────────────────┘
app = FastAPI(
    title="NeuroMotion API",
    description="""
    AI-powered digital healthcare platform for human movement intelligence.

    Features:
    - AI-powered yoga analysis
    - Posture correction
    - Gait analysis
    - Movement tracking
    - Personalized health insights

    Documentation:
    - API Docs: https://api.neuromation.ai/docs
    - Developer Portal: https://developers.neuromation.ai
    """,
    version=settings.APP_VERSION,
    openapi_url="/api/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ┌──────────────────────────────────────────────────────────────┐
# │ Middleware Configuration                                     │
# └──────────────────────────────────────────────────────────────┘

# CORS Configuration
if settings.ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS_LIST,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Add process time to response headers for performance monitoring.
    """
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    return response

# ┌──────────────────────────────────────────────────────────────┐
# │ Global Exception Handler                                     │
# └──────────────────────────────────────────────────────────────┘
@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    """
    Handle custom application exceptions.

    Returns consistent error responses across all endpoints.
    """
    logger.error(
        "Application error",
        error_code=exc.code,
        message=exc.message,
        path=request.url.path,
        method=request.method,
        detail=exc.detail,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.code,
                "message": exc.message,
                "detail": exc.detail,
            },
            "timestamp": time.time(),
        },
    )

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    Handle unhandled exceptions and prevent 500 errors from exposing
    internal details.
    """
    error_id = f"{int(time.time() * 1_000_000):x}"[-8:]

    logger.error(
        "Unhandled exception",
        error_id=error_id,
        path=request.url.path,
        method=request.method,
        exception=str(exc),
        traceback=str(exc.__traceback__),
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "error_id": error_id,
            },
            "timestamp": time.time(),
        },
    )

# ┌──────────────────────────────────────────────────────────────┐
# │ Routes                                                       │
# └──────────────────────────────────────────────────────────────┘
app.include_router(
    api_router,
    prefix=settings.API_V1_STR,
)

# ┌──────────────────────────────────────────────────────────────┐
# │ Health Check Endpoints                                       │
# └──────────────────────────────────────────────────────────────┘
@app.get("/health", tags=["health"])
async def health_check():
    """
    Basic health check endpoint for load balancers and monitoring.

    Returns 200 if the service is running.
    """
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "timestamp": time.time(),
    }

@app.get("/health/live", tags=["health"])
async def liveness_probe():
    """
    Kubernetes-style liveness probe.

    Use this endpoint for container restarts based on health.
    """
    return {"status": "alive"}

@app.get("/health/ready", tags=["health"])
async def readiness_probe():
    """
    Kubernetes-style readiness probe.

    Verifies that all dependencies (database, cache) are available.
    """
    checks = {
        "database": False,
        "redis": False,
    }

    # Check database
    try:
        from app.core.database import db
        async with db.engine.connect() as conn:
            await conn.execute(db.text("SELECT 1"))
        checks["database"] = True
    except Exception:
        checks["database"] = False

    # Check Redis
    try:
        from app.core.database import redis_client
        if redis_client:
            await redis_client.ping()
        checks["redis"] = True
    except Exception:
        checks["redis"] = False

    all_healthy = all(checks.values())
    status_code = status.HTTP_200_OK if all_healthy else status.HTTP_503_SERVICE_UNAVAILABLE

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "healthy" if all_healthy else "degraded",
            "checks": checks,
            "timestamp": time.time(),
        },
    )


# ┌──────────────────────────────────────────────────────────────┐
# │ Root Endpoint                                                │
# └──────────────────────────────────────────────────────────────┘
@app.get("/", tags=["root"])
async def root():
    """
    Root endpoint returning basic API information.

    Not intended for production use - serves as a simple entry point
    for health checks and service discovery.
    """
    return {
        "name": "NeuroMotion API",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "api_docs": "/redoc",
        "health": "/health",
        "api_version": settings.API_V1_STR,
    }
