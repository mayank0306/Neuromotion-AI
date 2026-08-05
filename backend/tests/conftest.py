"""
Test configuration and fixtures.

Provides common fixtures used across all test suites.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from typing import AsyncGenerator

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.main import app
from app.core.database import db, Database
from app.core.config import settings


# ┌─────────────────────────────────────────────────────────────────┐
# │ Fixtures                                                        │
# └─────────────────────────────────────────────────────────────────┘

@pytest.fixture(scope="session")
def test_database_url():
    """Create test database URL using SQLite."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield f"sqlite+aiosqlite:///{db_path}"
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture(scope="function")
async def test_db(test_database_url):
    """Create a fresh database for each test function."""
    engine = create_async_engine(
        test_database_url,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
    )
    
    # Create tables
    from app.models.base import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Create session factory
    async_session = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    # Override database dependency
    original_engine = db.engine
    original_session = db.AsyncSessionLocal
    
    db.engine = engine
    db.AsyncSessionLocal = async_session
    
    yield async_session
    
    # Restore original values
    db.engine = original_engine
    db.AsyncSessionLocal = original_session
    
    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest.fixture(scope="function")
async def client(test_db):
    """Create a test client with database override."""
    from app.core.database import get_db
    
    async def override_get_db():
        async_session = test_db
        async with async_session() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    app.dependency_overrides[get_db] = override_get_db
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as test_client:
        yield test_client
    
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
async def auth_headers(client):
    """Get authentication headers for test user."""
    # Register test user
    await client.post("/api/v1/auth/register", json={
        "email": "test@example.com",
        "password": "SecurePassword123!",
        "first_name": "Test",
        "last_name": "User",
    })
    
    # Login
    response = await client.post("/api/v1/auth/login", json={
        "email": "test@example.com",
        "password": "SecurePassword123!",
    })
    
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
def mock_ai_service():
    """Mock AI service for tests that don't require real inference."""
    # Can be overridden in specific tests
    pass


# ┌─────────────────────────────────────────────────────────────────┐
# │ Test Categories                                                 │
# └─────────────────────────────────────────────────────────────────┘

def pytest_configure(config):
    """Register additional pytest markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end tests"
    )
    config.addinivalue_line(
        "markers", "ai: AI-specific tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests"
    )


@pytest.fixture
def mock_pose_estimator():
    """Mock pose estimator for unit tests."""
    class MockPoseEstimator:
        async def initialize(self):
            pass
        
        async def estimate(self, image):
            # Return mock pose data (33 landmarks)
            return [
                (0.5, 0.5, 0.0, 0.9) for _ in range(33)
            ]
        
        def get_landmark_names(self):
            return ["nose", "left_shoulder", "right_shoulder"]
        
        async def close(self):
            pass
    
    return MockPoseEstimator()
