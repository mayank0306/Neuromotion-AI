"""API routes for the application."""

from fastapi import APIRouter

from app.api.v1 import api_router

__all__ = ["api_router"]
