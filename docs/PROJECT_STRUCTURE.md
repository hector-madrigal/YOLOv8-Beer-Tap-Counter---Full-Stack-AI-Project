# 🍺 Beer Counter System - Project Structure

## 📁 Directory Organization (Clean for Delivery)

### **Core System (Production)**
```
├── backend/              # FastAPI backend server
│   ├── app/             # Application logic
│   │   ├── main.py      # API endpoints
│   │   ├── yolo_video_processor.py  # YOLOv8 detection engine
│   │   ├── video_processor.py       # Legacy processor
│   │   ├── tap_detector.py          # Template matching
│   │   ├── roi_config.py            # ROI coordinates
│   │   ├── config.py                # Configuration
│   │   ├── models.py                # Database models
│   │   ├── database.py              # DB connection
│   │   ├── schemas.py               # API schemas
│   │   └── crud.py                  # Database operations
│   ├── uploads/         # Uploaded videos (empty for delivery)
│   ├── requirements.txt
│   ├── Dockerfile
│   └── beer_counter.db  # SQLite database
│
├── frontend/            # Web interface
│   ├── public/         # Static assets
│   │   ├── logo_gamb00za.png  # Brand logo
│   │   └── videos/     # Demo videos
│   │       ├── cerveza1.mp4
│   │       ├── cerveza2.mp4
│   │       ├── cerveza3.mp4
│   │       └── ... (8 total)
│   ├── index.html      # Main UI
│   └── app.js          # Frontend logic
│
├── experimental/        # Detection engine
│   └── realtime_cup_detector.py  # Main YOLOv8 detector
│
├── templates/          # Tap templates for detection
│   ├── tapA_up.png
│   └── tapB_up.png
│
└── runs/               # Trained model
    └── detect/
        └── train_corrected2/
            └── weights/
                └── best.pt  # YOLOv8 fine-tuned model
```

### **Development Tools (Optional - For Reference)**
```
├── datasets/           # Training data (can be removed if space needed)
│   └── beer_cups/     # YOLOv8 training dataset
│       ├── images/
│       └── labels/
│
├── tools/              # Configuration utilities
│   ├── configure_rois.py
│   ├── reconfigure_flow_rois.py
│   └── capture_templates.py
│
└── config/            # Additional configuration files
```

### **Documentation**
```
├── docs/              # Project documentation
│   ├── PROJECT_STRUCTURE.md  # This file
│   └── TECHNICAL.md          # Technical details
│
├── README.md          # Main project README
└── docker-compose.yml # Docker deployment config
```

---

## 🎯 Key System Components

### **Detection Pipeline**
1. **Video Upload** → `backend/app/main.py` (FastAPI endpoint)
2. **YOLOv8 Processing** → `backend/app/yolo_video_processor.py`
3. **Model Inference** → `runs/detect/train_corrected2/weights/best.pt`
4. **Centroid Tracking** → Object tracking with occlusion tolerance
5. **Tap Detection** → Template matching for tap identification
6. **Database Storage** → SQLite persistence

### **Model Details**
- **Base Model**: YOLOv8n
- **Fine-tuned**: Custom dataset (beer glasses)
- **Threshold**: 265 frames to qualify
- **Tap Validation**: 20 frames minimum
- **Occlusion Tolerance**: 150 frames

---

## 📊 Current System Features

1. ✅ **YOLOv8 Detection** - Fine-tuned beer glass detection
2. ✅ **Centroid Tracking** - Multi-object tracking with occlusion handling
3. ✅ **Tap Identification** - Template matching (left/right tap)
4. ✅ **Web Interface** - Upload and process videos
5. ✅ **Database Tracking** - SQLite beer event storage
6. ✅ **Docker Deployment** - Containerized application
7. ✅ **Brand Integration** - gamb00za logo and footer

---

## 🚀 How to Use

### **Run with Docker (Recommended):**
```bash
docker-compose up --build

# Access:
# - Frontend: http://localhost:8080
# - Backend API: http://localhost:8000/docs
```

### **Manual Setup:**
```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend
cd frontend
# Open index.html in browser or use:
python -m http.server 8080
```

### **Test Detection:**
```bash
# Use demo videos from frontend/public/videos/
# Upload via web interface at http://localhost:8080
```

---

## 📝 Files Removed for Delivery

**Cleaned up (~5.5 GB):**
- ❌ `venv/` - Virtual environments (recreate with requirements.txt)
- ❌ `.conda/` - Conda environments (recreate)
- ❌ `backend/.conda/` - Duplicate environment
- ❌ `backend/venv/` - Duplicate environment
- ❌ `backend/uploads/` - Old uploaded videos (empty folder maintained)
- ❌ `dev/` - Development scripts
- ❌ `experimental/*.py` - Kept only `realtime_cup_detector.py`
- ❌ `runs/detect/train*/` - Old training runs (kept only train_corrected2)
- ❌ `yolov8n.pt` - Base pretrained model (not needed)

**Essential Files Kept:**
- ✅ `runs/detect/train_corrected2/weights/best.pt` - Trained model
- ✅ `backend/app/` - All backend code
- ✅ `frontend/` - Web interface with logo and demo videos
- ✅ `templates/` - Tap detection templates
- ✅ `experimental/realtime_cup_detector.py` - Detection engine
- ✅ `backend/beer_counter.db` - Database (can be emptied if needed)
- ✅ `datasets/` - Training data (optional, can be removed if space needed)

---

## 🎓 Project Context

**Caso Práctico - Full Stack & AI Developer (gamb00za)**

This system demonstrates:
- YOLOv8 fine-tuning for custom object detection
- FastAPI backend with async video processing
- Docker containerization
- Centroid-based object tracking
- Template matching for classification
- SQLite persistence
- Modern web interface

**Optimizations Applied:**
- Reduced detection delay from 10s → 1s
- Frame threshold: 270 → 265 (99.6% accuracy maintained)
- Tap validation: 200 → 20 frames
- Occlusion tolerance: 150 frames
- Expanded search radius: 2% per frame, max 3x

**Why separate tools?**
- Clear separation of setup vs runtime
- Professional project organization
- Easy onboarding for new developers
- Maintenance clarity

**Why template matching over YOLO?**
- Simpler, faster, more reliable
- No training data needed
- Works with existing camera setup
- Lower computational requirements

---

## 🔄 Future Improvements

- [ ] Calibration wizard for new installations
- [ ] Advanced analytics dashboard
- [ ] Export pour reports (CSV/PDF)
- [ ] Multi-camera support
- [ ] Cloud deployment guides
