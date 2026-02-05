"""
Database configuration and session management.
SQLite database with Docker volume persistence.
"""
import os
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.config import DATABASE_URL


def ensure_db_directory():
    """Ensure the database directory exists (for Docker volume)."""
    if "sqlite" in DATABASE_URL:
        # Extract path from sqlite:///path or sqlite:////path (absolute)
        db_path = DATABASE_URL.replace("sqlite:///", "")
        if db_path.startswith("/"):
            # Absolute path (Docker: sqlite:////app/data/beer_counter.db)
            db_dir = Path(db_path).parent
        else:
            # Relative path
            db_dir = Path(db_path).parent
        
        if db_dir and str(db_dir) != ".":
            db_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Database directory ensured: {db_dir}")


def get_engine_args():
    """Get engine arguments for SQLite."""
    return {
        "connect_args": {"check_same_thread": False}
    }


# Ensure database directory exists before creating engine
ensure_db_directory()

# Create engine
engine = create_engine(DATABASE_URL, **get_engine_args())

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db():
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created/verified")
