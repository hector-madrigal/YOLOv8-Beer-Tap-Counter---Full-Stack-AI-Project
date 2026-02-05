"""
CRUD operations for the beer counter application.
"""
from datetime import datetime
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.models import Video, PourEvent, ProcessingStatus
from app.schemas import VideoCreate, PourEventCreate


# Video CRUD operations

def create_video(db: Session, video: VideoCreate) -> Video:
    """Create a new video record."""
    db_video = Video(
        filename=video.filename,
        original_filename=video.original_filename,
        file_path=video.file_path,
        status=ProcessingStatus.PENDING.value
    )
    db.add(db_video)
    db.commit()
    db.refresh(db_video)
    return db_video


def get_video(db: Session, video_id: int) -> Optional[Video]:
    """Get a video by ID."""
    return db.query(Video).filter(Video.id == video_id).first()


def get_videos(
    db: Session, 
    skip: int = 0, 
    limit: int = 100,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> List[Video]:
    """Get all videos with optional filtering."""
    query = db.query(Video)
    
    if start_date:
        query = query.filter(Video.created_at >= start_date)
    if end_date:
        query = query.filter(Video.created_at <= end_date)
    
    return query.order_by(Video.created_at.desc()).offset(skip).limit(limit).all()


def update_video_status(
    db: Session, 
    video_id: int, 
    status: ProcessingStatus,
    error_message: Optional[str] = None,
    duration: Optional[float] = None
) -> Optional[Video]:
    """Update video processing status."""
    db_video = get_video(db, video_id)
    if db_video:
        db_video.status = status.value
        if error_message:
            db_video.error_message = error_message
        if duration:
            db_video.duration_seconds = duration
        if status == ProcessingStatus.COMPLETED:
            db_video.processed_at = datetime.utcnow()
        db.commit()
        db.refresh(db_video)
    return db_video


def delete_video(db: Session, video_id: int) -> bool:
    """Delete a video and its associated pour events."""
    db_video = get_video(db, video_id)
    if db_video:
        db.delete(db_video)
        db.commit()
        return True
    return False


# Pour Event CRUD operations

def create_pour_event(db: Session, pour_event: PourEventCreate) -> PourEvent:
    """Create a new pour event."""
    db_event = PourEvent(
        video_id=pour_event.video_id,
        tap=pour_event.tap,
        start_frame=pour_event.start_frame,
        end_frame=pour_event.end_frame,
        start_time_seconds=pour_event.start_time_seconds,
        end_time_seconds=pour_event.end_time_seconds,
        confidence=pour_event.confidence
    )
    db.add(db_event)
    db.commit()
    db.refresh(db_event)
    return db_event


def create_pour_events_bulk(db: Session, pour_events: List[PourEventCreate]) -> List[PourEvent]:
    """Create multiple pour events in bulk."""
    db_events = [
        PourEvent(
            video_id=pe.video_id,
            tap=pe.tap,
            start_frame=pe.start_frame,
            end_frame=pe.end_frame,
            start_time_seconds=pe.start_time_seconds,
            end_time_seconds=pe.end_time_seconds,
            confidence=pe.confidence
        )
        for pe in pour_events
    ]
    db.add_all(db_events)
    db.commit()
    for event in db_events:
        db.refresh(event)
    return db_events


def get_pour_events_by_video(db: Session, video_id: int) -> List[PourEvent]:
    """Get all pour events for a video."""
    return db.query(PourEvent).filter(PourEvent.video_id == video_id).all()


def get_pour_events_by_tap(
    db: Session, 
    tap: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> List[PourEvent]:
    """Get all pour events for a specific tap."""
    query = db.query(PourEvent).join(Video).filter(PourEvent.tap == tap.upper())
    
    if start_date:
        query = query.filter(Video.created_at >= start_date)
    if end_date:
        query = query.filter(Video.created_at <= end_date)
    
    return query.all()


def get_count_summary(
    db: Session,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> dict:
    """Get count summary for all videos."""
    query = db.query(Video).filter(Video.status == ProcessingStatus.COMPLETED.value)
    
    if start_date:
        query = query.filter(Video.created_at >= start_date)
    if end_date:
        query = query.filter(Video.created_at <= end_date)
    
    videos = query.all()
    
    tap_a_total = sum(v.tap_a_count for v in videos)
    tap_b_total = sum(v.tap_b_count for v in videos)
    
    return {
        "tap_a_total": tap_a_total,
        "tap_b_total": tap_b_total,
        "grand_total": tap_a_total + tap_b_total,
        "video_count": len(videos)
    }
