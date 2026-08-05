"""Integration tests for authentication."""

import pytest
from httpx import AsyncClient


class TestAuthIntegration:
    """Integration tests for authentication endpoints."""
    
    @pytest.mark.asyncio
    async def test_user_registration(self, client: AsyncClient):
        response = await client.post("/api/v1/auth/register", json={
            "email": "newuser@test.com",
            "password": "SecurePass123!",
            "first_name": "New",
            "last_name": "User",
        })
        
        assert response.status_code == 200
        assert response.json()["user"]["email"] == "newuser@test.com"

    @pytest.mark.asyncio
    async def test_duplicate_registration_fails(self, client: AsyncClient):
        # Register first user
        await client.post("/api/v1/auth/register", json={
            "email": "duplicate@test.com",
            "password": "SecurePass123!",
            "first_name": "User",
            "last_name": "One",
        })
        
        # Try to register again
        response = await client.post("/api/v1/auth/register", json={
            "email": "duplicate@test.com",
            "password": "SecurePass456!",
            "first_name": "User",
            "last_name": "Two",
        })
        
        assert response.status_code == 409

    @pytest.mark.asyncio
    async def test_login_with_invalid_credentials(self, client: AsyncClient):
        response = await client.post("/api/v1/auth/login", json={
            "email": "nonexistent@test.com",
            "password": "SecurePass123!",
        })
        
        assert response.status_code == 401
