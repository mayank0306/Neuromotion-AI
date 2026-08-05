"""
Database configuration and session management.

Provides:
- Async SQLAlchemy engine and session management
- Redis connection management
- Database initialization utilities
"""

from typing import Optional, AsyncGenerator
import structlog

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
    async_engine_from_config,
)
from sqlalchemy.pool import NullPool
import redis.asyncio as redis

from app.core.config import settings

logger = structlog.get_logger(__name__)

# ┌──────────────────────────────────────────────────────────────┐
# │ Database Engine                                              │
# └──────────────────────────────────────────────────────────────┘
class Database:
    """
    Async database engine and session management.

    Provides a singleton pattern for database access with
    connection pooling and error handling.
    """

    def __init__(self):
        self.engine: Optional[AsyncEngine] = None
        self.AsyncSessionLocal: Optional[async_sessionmaker[AsyncSession]] = None
        self._initialized: bool = False

    async def initialize(self):
        """Initialize database engine and session factory."""
        if self._initialized:
            return

        logger.info("Initializing database connection...", url=str(settings.DATABASE_URL))

        try:
            self.engine = create_async_engine(
                settings.DATABASE_URL,
                pool_pre_ping=True,
                pool_recycle=300,
                pool_size=settings.DATABASE_POOL_SIZE,
                max_overflow=settings.DATABASE_MAX_OVERFLOW,
                echo=settings.APP_DEBUG,
            )

            self.AsyncSessionLocal = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
                autocommit=False,
            )

            self._initialized = True
            logger.info("Database connection initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize database", error=str(e))
            raise

    async def close(self):
        """Close database connections."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connections closed")

    async def get_session(self) -> AsyncSession:
        """Get a database session."""
        if not self.AsyncSessionLocal:
            raise RuntimeError("Database not initialized")
        return self.AsyncSessionLocal()


# Global database instance
db = Database()

# ┌──────────────────────────────────────────────────────────────┐
# │ Redis Client                                                 │
# └──────────────────────────────────────────────────────────────┘
class RedisClient:
    """
    Redis client for caching and message queuing.

    Provides connection management with error handling
    and connection pooling.
    """

    def __init__(self):
        self.client: Optional[redis.Redis] = None
        self._initialized: bool = False

    async def initialize(self):
        """Initialize Redis client."""
        if self._initialized:
            return

        logger.info("Initializing Redis connection...", url=settings.REDIS_URL)

        try:
            self.client = redis.from_url(
                settings.REDIS_URL,
                password=settings.REDIS_PASSWORD,
                decode_responses=True,
                retry_on_timeout=True,
                max_connections=20,
            )
            self._initialized = True
            logger.info("Redis connection initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize Redis", error=str(e))
            raise

    async def close(self):
        """Close Redis connections."""
        if self.client:
            await self.client.close()
            logger.info("Redis connections closed")

    async def get(self, key: str) -> Optional[str]:
        """Get value from Redis."""
        if not self.client:
            return None
        return await self.client.get(key)

    async def set(
        self,
        key: str,
        value: str,
        expire: Optional[int] = None,
    ) -> None:
        """Set value in Redis."""
        if not self.client:
            return
        await self.client.set(key, value, ex=expire)

    async def delete(self, key: str) -> int:
        """Delete key from Redis."""
        if not self.client:
            return 0
        return await self.client.delete(key)

    async def ping(self) -> bool:
        """Ping Redis to check connection."""
        if not self.client:
            return False
        return await self.client.ping()


# Global Redis client instance
redis_client = RedisClient() if not settings.ENABLE_TELEMEDICINE else None


# ┌──────────────────────────────────────────────────────────────┐
# │ Database Dependencies                                        │
# └──────────────────────────────────────────────────────────────┘
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database session.

    Usage:
        @app.get("/users/")
        async def read_users(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(User))
            return result.scalars().all()
    """
    if not db.AsyncSessionLocal:
        await db.initialize()

    async with db.AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_cache() -> redis.Redis:
    """
    FastAPI dependency for Redis cache.

    Usage:
        @app.get("/cached/")
        async def read_cached(cache: redis.Redis = Depends(get_cache)):
            cached = await cache.get("key")
            ...
    """
    if not redis_client:
        raise RuntimeError("Redis not initialized")
    if not redis_client.client:
        raise RuntimeError("Redis client not initialized")
    return redis_client.client


# ┌──────────────────────────────────────────────────────────────┐
# │ Initialization                                               │
# └──────────────────────────────────────────────────────────────┘
async def init_db():
    """
    Initialize database connections and create tables if needed.

    This function is called during application startup.
    """
    await db.initialize()

    # Initialize Redis if enabled
    if settings.REDIS_URL:
        try:
            await redis_client.initialize()
        except Exception as e:
            logger.warning("Redis initialization failed", error=str(e))

    logger.info("Database initialization complete")


async def close_db():
    """
    Close all database and Redis connections.

    This function is called during application shutdown.
    """
    await db.close()

    if redis_client:
        await redis_client.close()

    logger.info("Database connections closed")
