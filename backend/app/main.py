"""
FastAPI main application for the Beer Counter API.
"""
import os
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.database import get_db, init_db
from app.models import ProcessingStatus
from app.schemas import (
    VideoResponse, 
    VideoListResponse,
    ProcessingStatusResponse,
    CountSummary,
    PourEventCreate
)
from app.crud import (
    create_video,
    get_video,
    get_videos,
    update_video_status,
    delete_video,
    create_pour_events_bulk,
    get_count_summary
)
from app.yolo_video_processor import process_video_file
from app.config import UPLOAD_DIR, ALLOWED_EXTENSIONS

# Create FastAPI app
app = FastAPI(
    title="Beer Counter API",
    description="API for counting beer pours from video footage of a dual-tap system",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for video processing
executor = ThreadPoolExecutor(max_workers=2)


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    print("🚀 Starting Beer Counter API...")
    try:
        init_db()
        UPLOAD_DIR.mkdir(exist_ok=True)
        print("✅ Database initialized and upload directory created!")
    except Exception as e:
        print(f"❌ Startup error: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Beer Counter API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.post("/api/videos/upload", response_model=VideoResponse)
async def upload_video(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload a video file for processing.
    
    Accepts MP4 and MOV video formats.
    """
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Generate unique filename
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = UPLOAD_DIR / unique_filename
    
    # Save file
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Create database record
    from app.schemas import VideoCreate
    video_data = VideoCreate(
        filename=unique_filename,
        original_filename=file.filename,
        file_path=str(file_path)
    )
    
    db_video = create_video(db, video_data)
    
    return VideoResponse(
        id=db_video.id,
        filename=db_video.filename,
        original_filename=db_video.original_filename,
        status=db_video.status,
        error_message=db_video.error_message,
        duration_seconds=db_video.duration_seconds,
        created_at=db_video.created_at,
        processed_at=db_video.processed_at,
        tap_a_count=db_video.tap_a_count,
        tap_b_count=db_video.tap_b_count,
        total_count=db_video.total_count,
        pour_events=[]
    )


def process_video_task(video_id: int, file_path: str, db_url: str):
    """
    Background task to process a video.
    
    This runs in a separate thread to avoid blocking the main event loop.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    # Create new database session for this thread
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    
    try:
        # Update status to processing
        update_video_status(db, video_id, ProcessingStatus.PROCESSING)
        
        # Process the video with YOLO
        print(f"🎯 Processing video: {file_path}")
        result = process_video_file(file_path)
        
        # Create pour events
        pour_events = [
            PourEventCreate(
                video_id=video_id,
                tap=d.tap,
                start_frame=d.start_frame,
                end_frame=d.end_frame,
                start_time_seconds=d.start_time,
                end_time_seconds=d.end_time,
                confidence=d.confidence
            )
            for d in result.detections
        ]
        
        if pour_events:
            create_pour_events_bulk(db, pour_events)
            print(f"✅ Created {len(pour_events)} pour events")
        
        # Update video with results including timestamp
        from app.models import Video
        video = db.query(Video).filter(Video.id == video_id).first()
        if video:
            video.duration_seconds = result.duration
            video.video_timestamp = result.video_timestamp
            video.processed_at = datetime.utcnow()
            db.commit()
        
        # Update status to completed
        update_video_status(
            db, video_id, 
            ProcessingStatus.COMPLETED,
            duration=result.duration
        )
        
    except Exception as e:
        # Update status to error
        update_video_status(
            db, video_id,
            ProcessingStatus.ERROR,
            error_message=str(e)
        )
    finally:
        db.close()


@app.post("/api/videos/{video_id}/process", response_model=ProcessingStatusResponse)
async def process_video(
    video_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Start processing a video to count beer pours.
    
    The processing runs in the background. Use the status endpoint to check progress.
    """
    # Get video
    db_video = get_video(db, video_id)
    if not db_video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Check if already processing or completed
    if db_video.status == ProcessingStatus.PROCESSING.value:
        raise HTTPException(status_code=400, detail="Video is already being processed")
    
    # Get database URL for background task
    from app.config import DATABASE_URL
    
    # Start background processing
    background_tasks.add_task(
        process_video_task,
        video_id,
        db_video.file_path,
        DATABASE_URL
    )
    
    # Update status to pending (will be changed to processing by the task)
    update_video_status(db, video_id, ProcessingStatus.PROCESSING)
    db.refresh(db_video)
    
    return ProcessingStatusResponse(
        video_id=db_video.id,
        status=db_video.status,
        tap_a_count=db_video.tap_a_count,
        tap_b_count=db_video.tap_b_count,
        total_count=db_video.total_count,
        error_message=db_video.error_message
    )


@app.get("/api/videos/{video_id}/status", response_model=ProcessingStatusResponse)
async def get_video_status(
    video_id: int,
    db: Session = Depends(get_db)
):
    """
    Get the processing status of a video.
    """
    db_video = get_video(db, video_id)
    if not db_video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return ProcessingStatusResponse(
        video_id=db_video.id,
        status=db_video.status,
        tap_a_count=db_video.tap_a_count,
        tap_b_count=db_video.tap_b_count,
        total_count=db_video.total_count,
        error_message=db_video.error_message
    )


@app.get("/api/videos/{video_id}", response_model=VideoResponse)
async def get_video_details(
    video_id: int,
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a video including pour events.
    """
    db_video = get_video(db, video_id)
    if not db_video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return db_video


@app.get("/api/videos", response_model=List[VideoListResponse])
async def list_videos(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    """
    List all videos with optional date filtering.
    """
    videos = get_videos(db, skip=skip, limit=limit, start_date=start_date, end_date=end_date)
    return videos


@app.delete("/api/videos/{video_id}")
async def remove_video(
    video_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete a video and its associated data.
    """
    db_video = get_video(db, video_id)
    if not db_video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Delete the file
    file_path = Path(db_video.file_path)
    if file_path.exists():
        file_path.unlink()
    
    # Delete database record
    delete_video(db, video_id)
    
    return {"message": "Video deleted successfully"}


@app.get("/api/summary", response_model=CountSummary)
async def get_summary(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    """
    Get summary counts across all processed videos.
    """
    summary = get_count_summary(db, start_date=start_date, end_date=end_date)
    return CountSummary(**summary)


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
