"""
Pydantic schemas for API request/response validation.
"""
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel


class PourEventBase(BaseModel):
    """Base schema for pour events."""
    tap: str
    start_frame: int
    end_frame: int
    start_time_seconds: float
    end_time_seconds: float
    confidence: float = 1.0


class PourEventCreate(PourEventBase):
    """Schema for creating a pour event."""
    video_id: int


class PourEventResponse(PourEventBase):
    """Schema for pour event response."""
    id: int
    video_id: int
    created_at: datetime

    class Config:
        from_attributes = True


class VideoBase(BaseModel):
    """Base schema for videos."""
    original_filename: str


class VideoCreate(VideoBase):
    """Schema for creating a video record."""
    filename: str
    file_path: str


class VideoResponse(BaseModel):
    """Schema for video response."""
    id: int
    filename: str
    original_filename: str
    status: str
    error_message: Optional[str] = None
    duration_seconds: Optional[float] = None
    video_timestamp: Optional[str] = None
    created_at: datetime
    processed_at: Optional[datetime] = None
    tap_a_count: int
    tap_b_count: int
    total_count: int
    pour_events: List[PourEventResponse] = []

    class Config:
        from_attributes = True


class VideoListResponse(BaseModel):
    """Schema for video list response."""
    id: int
    filename: str
    original_filename: str
    status: str
    video_timestamp: Optional[str] = None
    created_at: datetime
    processed_at: Optional[datetime] = None
    tap_a_count: int
    tap_b_count: int
    total_count: int

    class Config:
        from_attributes = True


class ProcessingStatusResponse(BaseModel):
    """Schema for processing status response."""
    video_id: int
    status: str
    tap_a_count: int
    tap_b_count: int
    total_count: int
    error_message: Optional[str] = None


class CountSummary(BaseModel):
    """Schema for count summary."""
    tap_a_total: int
    tap_b_total: int
    grand_total: int
    video_count: int


class DateRangeQuery(BaseModel):
    """Schema for date range query."""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    tap: Optional[str] = None
