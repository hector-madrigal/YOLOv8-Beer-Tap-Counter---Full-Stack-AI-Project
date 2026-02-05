"""
YOLOv8 Video Processing Service for Beer Detection.

This module integrates the successful YOLOv8-based beer detection system
with the FastAPI backend for web interface processing.
"""
import cv2
import numpy as np
import os
import sys
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from ultralytics import YOLO

# Import ROI configuration and app config
from app.roi_config import get_roi, extract_roi
from app.config import YOLO_MODEL_PATH, YOLO_CONFIDENCE_THRESHOLD, TEMPLATES_DIR

# Try to import OCR library
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("⚠️ pytesseract not available - timestamp extraction will be limited")


@dataclass
class PourDetection:
    """Represents a detected beer pour event."""
    tap: str  # 'LEFT' or 'RIGHT' 
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    confidence: float = 1.0


@dataclass
class VideoAnalysisResult:
    """Complete video analysis result."""
    detections: List[PourDetection]
    duration: float
    video_timestamp: Optional[str] = None
    tap_a_count: int = 0
    tap_b_count: int = 0
    total_count: int = 0


class YOLOBeerDetector:
    """
    YOLOv8-based beer detector that integrates the successful detection system.
    """
    
    def __init__(self):
        # Load YOLOv8 model
        if not YOLO_MODEL_PATH.exists():
            raise FileNotFoundError(f"YOLO model not found at {YOLO_MODEL_PATH}")
        
        self.model = YOLO(str(YOLO_MODEL_PATH))
        self.conf_threshold = YOLO_CONFIDENCE_THRESHOLD
        self.iou_threshold = 0.5  # Same as realtime_cup_detector
        
        # Check if this is a fine-tuned model (custom) or pretrained COCO
        model_path_str = str(YOLO_MODEL_PATH)
        self.is_custom_model = 'train' in model_path_str or 'best.pt' in model_path_str
        
        if self.is_custom_model:
            # Fine-tuned model: class 0 = cup
            self.beer_class_ids = [0]
            print("🎯 Fine-tuned model detected - looking for class 0 (cup)")
        else:
            # COCO classes that might detect beer glasses
            self.beer_class_ids = [39, 40, 41, 42, 45]  # bottle, wine_glass, cup, fork, bowl
            print("🏷️  COCO model detected - looking for classes 39-45 (bottle, wine_glass, cup, etc.)")
        
        # Load tap templates
        self.tap_templates = self._load_tap_templates()
        
        # ROI configuration
        try:
            self.roi_flow_l = get_roi('ROI_FLOW_L')
            self.roi_flow_r = get_roi('ROI_FLOW_R')  
            self.roi_tap_l = get_roi('ROI_TAP_L')
            self.roi_tap_r = get_roi('ROI_TAP_R')
            print(f"✓ ROI Configuration loaded:")
            print(f"  FLOW_L: {self.roi_flow_l}")
            print(f"  FLOW_R: {self.roi_flow_r}")
            print(f"  TAP_L: {self.roi_tap_l}")
            print(f"  TAP_R: {self.roi_tap_r}")
        except:
            # Fallback ROIs - ORIGINALES que funcionan
            self.roi_flow_l = (1932, 1101, 162, 483)
            self.roi_flow_r = (2055, 1227, 180, 462)
            self.roi_tap_l = (2035, 749, 83, 198)
            self.roi_tap_r = (2141, 845, 93, 246)
            print("⚠️ Using fallback ROI configuration")
        
        # Statistics - Frame counters
        self.total_detections_l = 0
        self.total_detections_r = 0
        self.total_smart_servings_l = 0  # Frames with cup + active tap
        self.total_smart_servings_r = 0  # Frames with cup + active tap
        
        # State tracking for beer counting (265 frames minimum)
        self.min_serving_frames = 265
        self.serving_frames_l = 0
        self.serving_frames_r = 0
        self.beers_served_l = 0
        self.beers_served_r = 0
        self.min_tap_active_frames = 20  # Minimum frames tap must be consecutively active to validate beer (reduced from 200)
        
        # Tap activity tracking for validation
        self.tap_active_frames_l = 0  # Consecutive frames tap L has been active
        self.tap_active_frames_r = 0  # Consecutive frames tap R has been active
        
        # Object tracking for individual beer counting
        self.max_objects_seen_l = 0  # Maximum number of objects seen simultaneously in left
        self.max_objects_seen_r = 0  # Maximum number of objects seen simultaneously in right
        self.objects_qualified_l = 0  # Number of objects that have met 265 frame requirement (left)
        self.objects_qualified_r = 0  # Number of objects that have met 265 frame requirement (right)
        
        # Unique serving session tracking
        self.empty_frames_l = 0  # Frames without cup in FLOW_L (to detect session end)
        self.empty_frames_r = 0  # Frames without cup in FLOW_R (to detect session end)
        self.min_empty_frames = 60  # Minimum empty frames to consider session ended (3 seconds at 20 FPS)
        
        # Pour detection state
        self.current_pour_l = None
        self.current_pour_r = None
        self.detections = []
        
        # Frame counting
        self.frame_count = 0
        
        # Centroid tracking to handle occlusions
        self.tracked_objects_l = {}  # {object_id: {last_centroid, frames_seen, not_seen_frames, qualified}}
        self.tracked_objects_r = {}
        self.next_object_id_l = 0
        self.next_object_id_r = 0
        self.max_object_distance = 100  # Max pixels to associate with same object
        self.occlusion_tolerance = 150  # Frames to tolerate occlusion (7.5 seconds at 20 FPS) - increased for arm occlusions
        self.min_frames_to_qualify = 265  # Objects with 265+ frames count as beer
        
    def extract_timestamp_from_frame(self, frame):
        """Extract timestamp from video frame using ROI_TS."""
        try:
            # Extract timestamp ROI
            ts_roi = extract_roi(frame, 'ROI_TS')
            if ts_roi is None:
                return None
            
            # Preprocess for better OCR
            gray = cv2.cvtColor(ts_roi, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to get white text on black background
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Increase size for better OCR
            scale_factor = 2
            binary = cv2.resize(binary, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
            
            if TESSERACT_AVAILABLE:
                # Use tesseract OCR
                text = pytesseract.image_to_string(binary, config='--psm 7')
                text = text.strip()
                
                # Try to extract date/time pattern (DD/MM/YYYY HH:MM:SS or similar)
                # Common patterns: 04/02/2026 15:30:45, 2026-02-04 15:30:45, etc.
                patterns = [
                    r'\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}',  # DD/MM/YYYY HH:MM:SS
                    r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}',  # YYYY-MM-DD HH:MM:SS
                    r'\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}:\d{2}',  # DD-MM-YYYY HH:MM:SS
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, text)
                    if match:
                        return match.group(0)
                
                # If no pattern matches but we got text, return it
                if text and len(text) > 5:
                    return text
            else:
                # Fallback: simple pattern matching on pixel values
                # This is less accurate but works without tesseract
                pass
            
            return None
            
        except Exception as e:
            print(f"⚠️ Error extracting timestamp: {e}")
            return None
        
    def _load_tap_templates(self):
        """Load tap templates for closed state detection."""
        templates = {}
        
        # Load template for left tap (closed state)
        tap_l_path = TEMPLATES_DIR / "tapA_up.png"
        if tap_l_path.exists():
            templates['tap_l_closed'] = cv2.imread(str(tap_l_path), cv2.IMREAD_GRAYSCALE)
            print(f"✓ Loaded left tap template: {tap_l_path}")
        else:
            print(f"⚠ Warning: Left tap template not found: {tap_l_path}")
            templates['tap_l_closed'] = None
        
        # Load template for right tap (closed state)
        tap_r_path = TEMPLATES_DIR / "tapB_up.png"
        if tap_r_path.exists():
            templates['tap_r_closed'] = cv2.imread(str(tap_r_path), cv2.IMREAD_GRAYSCALE)
            print(f"✓ Loaded right tap template: {tap_r_path}")
        else:
            print(f"⚠ Warning: Right tap template not found: {tap_r_path}")
            templates['tap_r_closed'] = None
            
        return templates
        
    def match_tap_template(self, roi_image, template, threshold=0.8):
        """Compare ROI with template using normalized cross correlation with illumination normalization."""
        if template is None:
            return False, 0.0
        
        # Resize template to match ROI size if needed
        roi_h, roi_w = roi_image.shape
        template_resized = cv2.resize(template, (roi_w, roi_h))
        
        # Normalize illumination using histogram equalization
        roi_normalized = cv2.equalizeHist(roi_image)
        template_normalized = cv2.equalizeHist(template_resized)
        
        # Apply Gaussian blur to reduce noise
        roi_normalized = cv2.GaussianBlur(roi_normalized, (5, 5), 0)
        template_normalized = cv2.GaussianBlur(template_normalized, (5, 5), 0)
        
        # Perform template matching on normalized images
        result = cv2.matchTemplate(roi_normalized, template_normalized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        # Return True if similarity is above threshold (tap is closed)
        is_closed = max_val >= threshold
        return is_closed, max_val

    def detect_tap_activity(self, frame):
        """Detect if taps are active using template matching."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Extract ROI areas for taps
        x_l, y_l, w_l, h_l = self.roi_tap_l
        x_r, y_r, w_r, h_r = self.roi_tap_r
        
        roi_tap_l = gray[y_l:y_l+h_l, x_l:x_l+w_l]
        roi_tap_r = gray[y_r:y_r+h_r, x_r:x_r+w_r]
        
        # Check if taps are closed (match templates) - using perfected thresholds
        tap_l_closed, similarity_l = self.match_tap_template(
            roi_tap_l, self.tap_templates.get('tap_l_closed'), 0.85)  # Higher threshold for LEFT tap to avoid false positives
        tap_r_closed, similarity_r = self.match_tap_template(
            roi_tap_r, self.tap_templates.get('tap_r_closed'), 0.6)  # Standard threshold for RIGHT
        
        # Tap is active if it's NOT closed (doesn't match closed template)
        tap_l_active = not tap_l_closed
        tap_r_active = not tap_r_closed
        
        # Debug: print similarities every 100 frames
        if hasattr(self, 'frame_count') and self.frame_count % 100 == 0:
            print(f"Frame {self.frame_count}: Tap L similarity: {similarity_l:.3f}, closed: {tap_l_closed}, active: {tap_l_active} | Tap R similarity: {similarity_r:.3f}, closed: {tap_r_closed}, active: {tap_r_active}")
        
        return tap_l_active, tap_r_active, similarity_l, similarity_r
    
    def point_in_roi(self, point, roi):
        """Check if point (x, y) is inside ROI (x, y, w, h)."""
        px, py = point
        rx, ry, rw, rh = roi
        return rx <= px <= rx + rw and ry <= py <= ry + rh
    
    def euclidean_distance(self, p1, p2):
        """Calculate euclidean distance between two points."""
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    
    def update_tracked_objects(self, current_centroids, tracked_objects, next_id, side='L'):
        """
        Update tracked objects with current frame detections using centroid tracking.
        Handles occlusions by tolerating missing frames and expanding search radius.
        """
        matched_ids = set()
        new_next_id = next_id
        
        # Match current centroids with existing tracked objects
        for centroid_data in current_centroids:
            centroid = centroid_data['centroid']
            best_match_id = None
            best_distance = float('inf')
            
            # Find closest tracked object - prioritize objects that have been occluded
            for obj_id, obj_data in tracked_objects.items():
                if obj_id in matched_ids:
                    continue  # Already matched
                
                # Expand search radius for occluded objects (they may have moved)
                # The longer occluded, the larger the search radius
                occlusion_factor = 1 + (obj_data['not_seen_frames'] * 0.02)  # +2% per frame occluded
                effective_max_distance = self.max_object_distance * min(occlusion_factor, 3.0)  # Max 3x expansion
                
                dist = self.euclidean_distance(centroid, obj_data['last_centroid'])
                if dist < best_distance and dist < effective_max_distance:
                    best_distance = dist
                    best_match_id = obj_id
            
            if best_match_id is not None:
                # Update existing tracked object
                matched_ids.add(best_match_id)
                tracked_objects[best_match_id]['last_centroid'] = centroid
                tracked_objects[best_match_id]['frames_seen'] += 1
                tracked_objects[best_match_id]['not_seen_frames'] = 0
            else:
                # Create new tracked object
                tracked_objects[new_next_id] = {
                    'last_centroid': centroid,
                    'frames_seen': 1,
                    'not_seen_frames': 0,
                    'qualified': False,
                    'created_at_frame': self.frame_count
                }
                matched_ids.add(new_next_id)
                new_next_id += 1
        
        # Update not_seen_frames for unmatched objects (handle occlusions)
        removed_objects = []
        for obj_id in list(tracked_objects.keys()):
            if obj_id not in matched_ids:
                tracked_objects[obj_id]['not_seen_frames'] += 1
                # Remove object if not seen for too long (definitive disappearance)
                if tracked_objects[obj_id]['not_seen_frames'] > self.occlusion_tolerance:
                    removed_objects.append((obj_id, tracked_objects[obj_id].copy()))
                    del tracked_objects[obj_id]
        
        return new_next_id, matched_ids, removed_objects
        
    def process_frame(self, frame, fps=20.0):
        """Process a single frame for beer detection with centroid tracking."""
        self.frame_count += 1
        
        # Detect tap activity using template matching
        tap_l_active, tap_r_active, similarity_l, similarity_r = self.detect_tap_activity(frame)
        
        # Run YOLO inference
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
        
        # Collect centroids for tracking
        current_centroids_l = []
        current_centroids_r = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()  # x1, y1, x2, y2
                conf = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                
                # Only process beer/cup related classes
                if class_id not in self.beer_class_ids:
                    continue
                
                # Get center point of detection
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Check which FLOW ROI the detection is in
                in_flow_l = self.point_in_roi((center_x, center_y), self.roi_flow_l)
                in_flow_r = self.point_in_roi((center_x, center_y), self.roi_flow_r)
                
                if in_flow_l:
                    current_centroids_l.append({
                        'centroid': (center_x, center_y),
                        'conf': conf,
                        'bbox': bbox
                    })
                    self.total_detections_l += 1
                    
                elif in_flow_r:
                    current_centroids_r.append({
                        'centroid': (center_x, center_y),
                        'conf': conf,
                        'bbox': bbox
                    })
                    self.total_detections_r += 1
        
        # Update tracked objects with centroid tracking
        self.next_object_id_l, matched_l, removed_l = self.update_tracked_objects(
            current_centroids_l, self.tracked_objects_l, self.next_object_id_l, 'L')
        self.next_object_id_r, matched_r, removed_r = self.update_tracked_objects(
            current_centroids_r, self.tracked_objects_r, self.next_object_id_r, 'R')
        
        # Check for removed objects that should count as beers (had enough frames but disappeared)
        for obj_id, obj_data in removed_l:
            if not obj_data['qualified'] and obj_data['frames_seen'] >= self.min_frames_to_qualify and self.tap_active_frames_l >= self.min_tap_active_frames:
                self.beers_served_l += 1
                print(f"BEER #{self.beers_served_l} SERVED - LEFT (object {obj_id} disappeared after {obj_data['frames_seen']} frames, tap active {self.tap_active_frames_l} frames)")
                # Create pour detection event
                start_frame = obj_data.get('created_at_frame', self.frame_count - obj_data['frames_seen'])
                detection = PourDetection(
                    tap='A',  # LEFT = A
                    start_frame=start_frame,
                    end_frame=self.frame_count,
                    start_time=start_frame / fps if fps > 0 else 0,
                    end_time=self.frame_count / fps if fps > 0 else 0,
                    confidence=1.0
                )
                self.detections.append(detection)
        
        for obj_id, obj_data in removed_r:
            if not obj_data['qualified'] and obj_data['frames_seen'] >= self.min_frames_to_qualify and self.tap_active_frames_r >= self.min_tap_active_frames:
                self.beers_served_r += 1
                print(f"BEER #{self.beers_served_r} SERVED - RIGHT (object {obj_id} disappeared after {obj_data['frames_seen']} frames, tap active {self.tap_active_frames_r} frames)")
                # Create pour detection event
                start_frame = obj_data.get('created_at_frame', self.frame_count - obj_data['frames_seen'])
                detection = PourDetection(
                    tap='B',  # RIGHT = B
                    start_frame=start_frame,
                    end_frame=self.frame_count,
                    start_time=start_frame / fps if fps > 0 else 0,
                    end_time=self.frame_count / fps if fps > 0 else 0,
                    confidence=1.0
                )
                self.detections.append(detection)
        
        # Count UNIQUE tracked objects (not just current frame detections)
        # Only count objects that have been seen for at least 5 frames (not noise)
        unique_cups_l = len([o for o in self.tracked_objects_l.values() if o['frames_seen'] >= 5])
        unique_cups_r = len([o for o in self.tracked_objects_r.values() if o['frames_seen'] >= 5])
        
        # Track tap activity for validation
        if tap_l_active:
            self.tap_active_frames_l += 1
        else:
            self.tap_active_frames_l = 0
            
        if tap_r_active:
            self.tap_active_frames_r += 1
        else:
            self.tap_active_frames_r = 0
        
        # Process LEFT tap (only count if tap has been active long enough)
        if tap_l_active and len(current_centroids_l) > 0 and self.tap_active_frames_l >= self.min_tap_active_frames:
            self.total_smart_servings_l += 1
            self.empty_frames_l = 0
            self.serving_frames_l += 1
            
            # Track maximum unique objects seen
            if unique_cups_l > self.max_objects_seen_l:
                self.max_objects_seen_l = unique_cups_l
            
            # Check for qualified beers (objects with enough frames)
            if self.serving_frames_l >= self.min_frames_to_qualify:
                # Count objects that have been tracked long enough AND not yet counted
                for obj_id, obj_data in self.tracked_objects_l.items():
                    if obj_data['frames_seen'] >= self.min_frames_to_qualify and not obj_data['qualified']:
                        obj_data['qualified'] = True
                        self.beers_served_l += 1
                        print(f"BEER #{self.beers_served_l} SERVED - LEFT (object {obj_id} qualified with {obj_data['frames_seen']} frames, tap active {self.tap_active_frames_l} frames)")
                        # Create pour detection event
                        start_frame = obj_data.get('created_at_frame', self.frame_count - obj_data['frames_seen'])
                        detection = PourDetection(
                            tap='A',  # LEFT = A
                            start_frame=start_frame,
                            end_frame=self.frame_count,
                            start_time=start_frame / fps if fps > 0 else 0,
                            end_time=self.frame_count / fps if fps > 0 else 0,
                            confidence=1.0
                        )
                        self.detections.append(detection)
        
        # Process RIGHT tap (only count if tap has been active long enough)
        if tap_r_active and len(current_centroids_r) > 0 and self.tap_active_frames_r >= self.min_tap_active_frames:
            self.total_smart_servings_r += 1
            self.empty_frames_r = 0
            self.serving_frames_r += 1
            
            # Track maximum unique objects seen
            if unique_cups_r > self.max_objects_seen_r:
                self.max_objects_seen_r = unique_cups_r
            
            # Check for qualified beers
            if self.serving_frames_r >= self.min_frames_to_qualify:
                for obj_id, obj_data in self.tracked_objects_r.items():
                    if obj_data['frames_seen'] >= self.min_frames_to_qualify and not obj_data['qualified']:
                        obj_data['qualified'] = True
                        self.beers_served_r += 1
                        print(f"BEER #{self.beers_served_r} SERVED - RIGHT (object {obj_id} qualified with {obj_data['frames_seen']} frames, tap active {self.tap_active_frames_r} frames)")
                        # Create pour detection event
                        start_frame = obj_data.get('created_at_frame', self.frame_count - obj_data['frames_seen'])
                        detection = PourDetection(
                            tap='B',  # RIGHT = B
                            start_frame=start_frame,
                            end_frame=self.frame_count,
                            start_time=start_frame / fps if fps > 0 else 0,
                            end_time=self.frame_count / fps if fps > 0 else 0,
                            confidence=1.0
                        )
                        self.detections.append(detection)
        
        # Handle end of serving session (no cups detected for extended period)
        if len(current_centroids_l) == 0:
            self.empty_frames_l += 1
            # Only reset if ALL tracked objects have disappeared
            if len(self.tracked_objects_l) == 0 and self.serving_frames_l > 0:
                print(f"Serving ended - LEFT (all objects gone, had {self.serving_frames_l} frames, served {self.beers_served_l} beers)")
                self.serving_frames_l = 0
                self.max_objects_seen_l = 0
        else:
            self.empty_frames_l = 0
            
        if len(current_centroids_r) == 0:
            self.empty_frames_r += 1
            if len(self.tracked_objects_r) == 0 and self.serving_frames_r > 0:
                print(f"Serving ended - RIGHT (all objects gone, had {self.serving_frames_r} frames, served {self.beers_served_r} beers)")
                self.serving_frames_r = 0
                self.max_objects_seen_r = 0
        else:
            self.empty_frames_r = 0
        
        return {
            'cups_l': len(current_centroids_l),
            'cups_r': len(current_centroids_r),
            'unique_cups_l': unique_cups_l,
            'unique_cups_r': unique_cups_r,
            'tap_l_active': tap_l_active,
            'tap_r_active': tap_r_active,
            'beers_l': self.beers_served_l,
            'beers_r': self.beers_served_r
        }


def process_video_file(video_path: str) -> VideoAnalysisResult:
    """
    Process a video file and return beer detection results.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        VideoAnalysisResult with detected pour events
    """
    print(f"🎬 Processing video: {video_path}")
    
    # Initialize detector
    detector = YOLOBeerDetector()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"📊 Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s duration")
    
    # Extract timestamp from first frame
    video_timestamp = None
    ret, first_frame = cap.read()
    if ret:
        video_timestamp = detector.extract_timestamp_from_frame(first_frame)
        if video_timestamp:
            print(f"📅 Video timestamp: {video_timestamp}")
        else:
            print("⚠️ Could not extract video timestamp")
        # Reset to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Process frame
            result = detector.process_frame(frame, fps)
            
            # Progress logging every 1000 frames
            if frame_count % 1000 == 0:
                print(f"📈 Processed {frame_count}/{total_frames} frames "
                      f"(Beers: L:{result['beers_l']} R:{result['beers_r']})")
        
    except Exception as e:
        print(f"❌ Error processing frame {frame_count}: {e}")
        raise
    finally:
        cap.release()
    
    # Convert tap names for compatibility
    for detection in detector.detections:
        if detection.tap == 'LEFT':
            detection.tap = 'A'
        elif detection.tap == 'RIGHT':
            detection.tap = 'B'
    
    print(f"🍺 Processing complete:")
    print(f"   Left tap beers: {detector.beers_served_l}")
    print(f"   Right tap beers: {detector.beers_served_r}")
    print(f"   Total detections: {len(detector.detections)}")
    if video_timestamp:
        print(f"   Video timestamp: {video_timestamp}")
    
    return VideoAnalysisResult(
        detections=detector.detections,
        duration=duration,
        video_timestamp=video_timestamp,
        tap_a_count=detector.beers_served_l,  # Left = A
        tap_b_count=detector.beers_served_r,  # Right = B  
        total_count=detector.beers_served_l + detector.beers_served_r
    )