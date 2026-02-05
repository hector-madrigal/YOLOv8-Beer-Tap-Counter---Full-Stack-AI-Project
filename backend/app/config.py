"""
Application configuration settings.
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Database
# SQLite with Docker volume persistence or local development
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    f"sqlite:///{BASE_DIR}/beer_counter.db"
)

# Log database type on startup
print(f"📦 Using SQLite database: {DATABASE_URL}")

# Upload settings
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Allowed video extensions
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

# Video processing settings
FRAME_SKIP = 5  # Process every Nth frame for performance
MOTION_THRESHOLD = 25  # Threshold for motion detection
MIN_POUR_DURATION_FRAMES = 15  # Minimum frames to consider a valid pour

# YOLO model settings
YOLO_MODEL_PATH = BASE_DIR / "runs" / "detect" / "train_corrected2" / "weights" / "best.pt"
YOLO_CONFIDENCE_THRESHOLD = 0.25  # Same as realtime_cup_detector default
TEMPLATES_DIR = BASE_DIR / "templates"
