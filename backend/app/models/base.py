"""Base model class with common fields."""

from datetime import datetime
from sqlalchemy import Column, DateTime, Uuid
from sqlalchemy.orm import declarative_base
import uuid

Base = declarative_base()

class BaseModel(Base):
    """Base model with common fields."""
    __abstract__ = True

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def to_dict(self) -> dict:
        """Convert model to dictionary."""
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }
