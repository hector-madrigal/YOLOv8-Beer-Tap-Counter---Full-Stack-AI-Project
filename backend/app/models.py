"""
SQLAlchemy models for the beer counter application.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, Enum
from sqlalchemy.orm import relationship
import enum

from app.database import Base


class ProcessingStatus(str, enum.Enum):
    """Status of video processing."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


class Video(Base):
    """Model for uploaded videos."""
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    status = Column(String(50), default=ProcessingStatus.PENDING.value)
    error_message = Column(String(1000), nullable=True)
    duration_seconds = Column(Float, nullable=True)
    video_timestamp = Column(String(100), nullable=True)  # Timestamp extracted from video
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    
    # Relationship to pour events
    pour_events = relationship("PourEvent", back_populates="video", cascade="all, delete-orphan")
    
    @property
    def tap_a_count(self) -> int:
        """Count of pours from Tap A."""
        return len([e for e in self.pour_events if e.tap == "A"])
    
    @property
    def tap_b_count(self) -> int:
        """Count of pours from Tap B."""
        return len([e for e in self.pour_events if e.tap == "B"])
    
    @property
    def total_count(self) -> int:
        """Total pour count."""
        return len(self.pour_events)


class PourEvent(Base):
    """Model for individual beer pour events."""
    __tablename__ = "pour_events"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    tap = Column(String(1), nullable=False)  # 'A' or 'B'
    start_frame = Column(Integer, nullable=False)
    end_frame = Column(Integer, nullable=False)
    start_time_seconds = Column(Float, nullable=False)
    end_time_seconds = Column(Float, nullable=False)
    confidence = Column(Float, default=1.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to video
    video = relationship("Video", back_populates="pour_events")
