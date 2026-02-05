#!/usr/bin/env python3
"""
YOLOv8 Real-time Cup Detection with ROI Visualization
Shows live detection of cups in beer flow ROIs for debugging and adjustment.
"""

import cv2
import argparse
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import sys
import os

# Load tap closed templates for matching
def load_tap_templates():
    templates = {}
    template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
    
    # Load template for left tap (closed state)
    tap_l_path = os.path.join(template_dir, "tapA_up.png")
    if os.path.exists(tap_l_path):
        templates['tap_l_closed'] = cv2.imread(tap_l_path, cv2.IMREAD_GRAYSCALE)
        print(f"✓ Loaded left tap template: {tap_l_path}")
    else:
        print(f"⚠ Warning: Left tap template not found: {tap_l_path}")
        templates['tap_l_closed'] = None
    
    # Load template for right tap (closed state)
    tap_r_path = os.path.join(template_dir, "tapB_up.png")
    if os.path.exists(tap_r_path):
        templates['tap_r_closed'] = cv2.imread(tap_r_path, cv2.IMREAD_GRAYSCALE)
        print(f"✓ Loaded right tap template: {tap_r_path}")
    else:
        print(f"⚠ Warning: Right tap template not found: {tap_r_path}")
        templates['tap_r_closed'] = None
    
    return templates

# Add backend/app to path to import roi_config
# Script está en /experimental/, necesita subir un nivel
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend', 'app'))
try:
    from roi_config import get_roi
    USE_EXISTING_ROIS = True
    print("✓ Using existing ROI configuration from roi_config.py")
except ImportError:
    print("⚠️ Warning: Could not import roi_config, using default ROIs")
    USE_EXISTING_ROIS = False

# Constantes de configuración
# Template matching threshold for tap state detection
TEMPLATE_THRESHOLD = 0.6  # Similarity threshold to consider tap closed


class RealTimeCupDetector:
    def __init__(self, model_path='runs/detect/train_corrected2/weights/best.pt', conf_threshold=0.25, iou_threshold=0.5):
        """
        Initialize real-time beer cup detector with tap flow detection
        """
        self.model = YOLO(model_path)
        print(f"📋 Using model: {model_path}")
        
        # Load tap templates for closed state detection
        self.tap_templates = load_tap_templates()
        self.template_threshold = 0.6  # Default threshold, but used per tap
        
        # Check if this is a fine-tuned model (custom) or pretrained COCO
        self.is_custom_model = 'train' in model_path or 'best.pt' in model_path
        
        if self.is_custom_model:
            # Fine-tuned model: class 0 = cup
            self.beer_class_ids = [0]
            print("🎯 Fine-tuned model detected - looking for class 0 (cup)")
        else:
            # COCO classes that might detect beer glasses
            self.beer_class_ids = [39, 40, 41, 42, 45]  # bottle, wine_glass, cup, fork, bowl
            print("🏷️  COCO model detected - looking for classes 39-45 (bottle, wine_glass, cup, etc.)")
        
        # Let's try ALL classes to see what it detects
        self.show_all_detections = False  # Start with beer-only
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # COCO class names for reference
        self.class_names = {
            0: 'person', 39: 'bottle', 40: 'wine_glass', 41: 'cup', 42: 'fork', 
            43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple'
        }
        
        # Load ROI configurations
        if USE_EXISTING_ROIS:
            self.roi_flow_l = get_roi('ROI_FLOW_L')  # Detection area for cups
            self.roi_flow_r = get_roi('ROI_FLOW_R')  # Detection area for cups
            self.roi_tap_l = get_roi('ROI_TAP_L')    # Tap flow detection
            self.roi_tap_r = get_roi('ROI_TAP_R')    # Tap flow detection
            
            print(f"ROI_FLOW_L: {self.roi_flow_l}")
            print(f"ROI_FLOW_R: {self.roi_flow_r}")
            print(f"ROI_TAP_L: {self.roi_tap_l}")
            print(f"ROI_TAP_R: {self.roi_tap_r}")
        else:
            # Fallback ROIs
            self.roi_flow_l = (1920, 1101, 172, 483)  # Valores originales que funcionan
            self.roi_flow_r = (2055, 1227, 180, 462)
            self.roi_tap_l = (2035, 749, 83, 198)
            self.roi_tap_r = (2141, 845, 93, 246)
        
        print(f"Confidence threshold: {conf_threshold}")
        print(f"Smart serving detection: Cup in FLOW + Tap activity required")
        print(f"Looking for classes: {self.beer_class_ids} (bottle, wine_glass, cup, fork, bowl)")
        
        # Statistics - Frame counters
        self.total_detections_l = 0
        self.total_detections_r = 0
        self.total_smart_servings_l = 0  # Frames with cup + active tap
        self.total_smart_servings_r = 0  # Frames with cup + active tap
        self.frame_count = 0
        
        # Statistics - Beer serving events (unique services)
        self.beers_served_l = 0  # Actual beers served (events)
        self.beers_served_r = 0  # Actual beers served (events)
        
        # State tracking for serving events
        self.serving_active_l = False  # Is left tap currently serving?
        self.serving_active_r = False  # Is right tap currently serving?
        
        # Object tracking for individual beer counting
        self.max_objects_seen_l = 0  # Maximum number of objects seen simultaneously in left
        self.max_objects_seen_r = 0  # Maximum number of objects seen simultaneously in right
        self.objects_qualified_l = 0  # Number of objects that have met 270 frame requirement (left)
        self.objects_qualified_r = 0  # Number of objects that have met 270 frame requirement (right)
        
        # Unique serving session tracking
        self.empty_frames_l = 0  # Frames without cup in FLOW_L (to detect session end)
        self.empty_frames_r = 0  # Frames without cup in FLOW_R (to detect session end)
        self.min_empty_frames = 60  # Minimum empty frames to consider session ended (3 seconds at 20 FPS)
        
        # Frame counting for minimum serving duration (265+ frames)
        self.serving_frames_l = 0  # Consecutive frames with cup in FLOW_L AND active tap
        self.serving_frames_r = 0  # Consecutive frames with cup in FLOW_R AND active tap
        self.min_serving_frames = 265  # Minimum frames to count as valid beer serving
        self.min_tap_active_frames = 20  # Minimum frames tap must be consecutively active to validate beer (reduced from 200)
        
        # Tap activity tracking for validation
        self.tap_active_frames_l = 0  # Consecutive frames tap L has been active
        self.tap_active_frames_r = 0  # Consecutive frames tap R has been active
        
        # Centroid tracking to handle occlusions
        self.tracked_objects_l = {}  # {object_id: {last_centroid, frames_seen, not_seen_frames, qualified}}
        self.tracked_objects_r = {}
        self.next_object_id_l = 0
        self.next_object_id_r = 0
        self.max_object_distance = 100  # Max pixels to associate with same object
        self.occlusion_tolerance = 150  # Frames to tolerate occlusion (7.5 seconds at 20 FPS) - increased for arm occlusions
        self.min_frames_to_qualify = 265  # Objects with 265+ frames count as beer

    def match_tap_template(self, roi_image, template, threshold=0.8):
        """Compare ROI with template using normalized cross correlation with illumination normalization"""
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
        """Detect if taps are active using template matching"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Extract ROI areas for taps
        x_l, y_l, w_l, h_l = self.roi_tap_l
        x_r, y_r, w_r, h_r = self.roi_tap_r
        
        roi_tap_l = gray[y_l:y_l+h_l, x_l:x_l+w_l]
        roi_tap_r = gray[y_r:y_r+h_r, x_r:x_r+w_r]
        
        # Check if taps are closed (match templates)
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
        """Check if point (x, y) is inside ROI (x, y, w, h)"""
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
    
    def draw_rois(self, frame):
        """Draw all ROI rectangles on frame"""
        # Draw FLOW ROIs (main detection areas)
        x, y, w, h = self.roi_flow_l
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Green
        cv2.putText(frame, 'FLOW L', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        x, y, w, h = self.roi_flow_r
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)  # Blue
        cv2.putText(frame, 'FLOW R', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Draw TAP ROIs (reference)
        x, y, w, h = self.roi_tap_l
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow
        cv2.putText(frame, 'TAP L', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw TAP ROIs with activity indicators
        x, y, w, h = self.roi_tap_l
        tap_l_color = (0, 255, 255) if hasattr(self, '_tap_l_active') and self._tap_l_active else (128, 128, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), tap_l_color, 2)
        status_l = "ACTIVE" if hasattr(self, '_tap_l_active') and self._tap_l_active else "IDLE"
        cv2.putText(frame, f'TAP L {status_l}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tap_l_color, 2)
        
        x, y, w, h = self.roi_tap_r
        tap_r_color = (255, 255, 0) if hasattr(self, '_tap_r_active') and self._tap_r_active else (128, 128, 128)
        cv2.rectangle(frame, (x, y), (x + w, y + h), tap_r_color, 2)
        status_r = "ACTIVE" if hasattr(self, '_tap_r_active') and self._tap_r_active else "IDLE"
        cv2.putText(frame, f'TAP R {status_r}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tap_r_color, 2)

    def process_frame(self, frame):
        """Process single frame and return annotated frame with centroid tracking"""
        self.frame_count += 1
        
        # Detect tap activity using template matching
        tap_l_active, tap_r_active, similarity_l, similarity_r = self.detect_tap_activity(frame)
        self._tap_l_active = tap_l_active  # Store for drawing
        self._tap_r_active = tap_r_active
        self._similarity_l = similarity_l  # Store similarity scores
        self._similarity_r = similarity_r
        
        # Run YOLO inference
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
        
        # Collect centroids for tracking
        current_centroids_l = []
        current_centroids_r = []
        all_detections = []  # For visualization
        
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
                
                # Get class name
                class_name = self.class_names.get(class_id, f"class_{class_id}")
                
                # Check which FLOW ROI the detection is in
                in_flow_l = self.point_in_roi((center_x, center_y), self.roi_flow_l)
                in_flow_r = self.point_in_roi((center_x, center_y), self.roi_flow_r)
                
                detection_info = {
                    'bbox': bbox,
                    'conf': conf,
                    'class_name': class_name,
                    'center': (center_x, center_y),
                    'in_flow_l': in_flow_l,
                    'in_flow_r': in_flow_r
                }
                all_detections.append(detection_info)
                
                if in_flow_l:
                    current_centroids_l.append({
                        'centroid': (center_x, center_y),
                        'conf': conf,
                        'bbox': bbox,
                        'class_name': class_name
                    })
                    self.total_detections_l += 1
                    
                elif in_flow_r:
                    current_centroids_r.append({
                        'centroid': (center_x, center_y),
                        'conf': conf,
                        'bbox': bbox,
                        'class_name': class_name
                    })
                    self.total_detections_r += 1
        
        # Update tracked objects with centroid tracking
        self.next_object_id_l, matched_l, removed_l = self.update_tracked_objects(
            current_centroids_l, self.tracked_objects_l, self.next_object_id_l, 'L')
        self.next_object_id_r, matched_r, removed_r = self.update_tracked_objects(
            current_centroids_r, self.tracked_objects_r, self.next_object_id_r, 'R')
        
        # Check for removed objects that should count as beers (had enough frames but disappeared)
        for obj_id, obj_data in removed_l:
            # print(f"[DEBUG] Object {obj_id} REMOVED - frames_seen: {obj_data['frames_seen']}, qualified: {obj_data['qualified']}, tap_active_frames: {self.tap_active_frames_l}")
            if not obj_data['qualified'] and obj_data['frames_seen'] >= self.min_frames_to_qualify and self.tap_active_frames_l >= self.min_tap_active_frames:
                self.beers_served_l += 1
                print(f"BEER #{self.beers_served_l} SERVED - LEFT (object {obj_id} disappeared after {obj_data['frames_seen']} frames, tap active {self.tap_active_frames_l} frames)")
        
        for obj_id, obj_data in removed_r:
            # print(f"[DEBUG] Object {obj_id} REMOVED - frames_seen: {obj_data['frames_seen']}, qualified: {obj_data['qualified']}, tap_active_frames: {self.tap_active_frames_r}")
            if not obj_data['qualified'] and obj_data['frames_seen'] >= self.min_frames_to_qualify and self.tap_active_frames_r >= self.min_tap_active_frames:
                self.beers_served_r += 1
                print(f"BEER #{self.beers_served_r} SERVED - RIGHT (object {obj_id} disappeared after {obj_data['frames_seen']} frames, tap active {self.tap_active_frames_r} frames)")
        
        # Count UNIQUE tracked objects (not just current frame detections)
        unique_cups_l = len([o for o in self.tracked_objects_l.values() if o['frames_seen'] >= 1])
        unique_cups_r = len([o for o in self.tracked_objects_r.values() if o['frames_seen'] >= 1])
        
        smart_servings_l = 0
        smart_servings_r = 0
        
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
            smart_servings_l = 1
            self.total_smart_servings_l += 1
            self.empty_frames_l = 0
            self.serving_frames_l += 1
            
            # Track maximum unique objects seen
            if unique_cups_l > self.max_objects_seen_l:
                self.max_objects_seen_l = unique_cups_l
            
            # Check for qualified beers (objects with enough frames)
            if self.serving_frames_l >= self.min_frames_to_qualify:
                for obj_id, obj_data in self.tracked_objects_l.items():
                    if obj_data['frames_seen'] >= self.min_frames_to_qualify and not obj_data['qualified']:
                        obj_data['qualified'] = True
                        self.beers_served_l += 1
                        print(f"BEER #{self.beers_served_l} SERVED - LEFT (object {obj_id} qualified with {obj_data['frames_seen']} frames, tap active {self.tap_active_frames_l} frames)")
        
        # Process RIGHT tap (only count if tap has been active long enough)
        if tap_r_active and len(current_centroids_r) > 0 and self.tap_active_frames_r >= self.min_tap_active_frames:
            smart_servings_r = 1
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
        
        # Handle end of serving session
        if len(current_centroids_l) == 0:
            self.empty_frames_l += 1
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
        
        # Draw detections with tracking info
        for det in all_detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox.astype(int)
            center_x, center_y = det['center']
            
            if det['in_flow_l']:
                if tap_l_active:
                    color = (0, 255, 0)  # Bright green
                    label = f"TRACK L ({self.serving_frames_l}f, {unique_cups_l} obj): {det['class_name']} {det['conf']:.2f}"
                else:
                    color = (0, 200, 0)
                    label = f"CUP L (no tap): {det['class_name']} {det['conf']:.2f}"
            elif det['in_flow_r']:
                if tap_r_active:
                    color = (255, 0, 0)  # Bright blue
                    label = f"TRACK R ({self.serving_frames_r}f, {unique_cups_r} obj): {det['class_name']} {det['conf']:.2f}"
                else:
                    color = (200, 0, 0)
                    label = f"CUP R (no tap): {det['class_name']} {det['conf']:.2f}"
            else:
                color = (0, 255, 255)  # Yellow
                label = f"{det['class_name']} {det['conf']:.2f}"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (int(center_x), int(center_y)), 5, color, -1)
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw ROIs
        self.draw_rois(frame)
        
        # Add statistics overlay
        stats_text = [
            f"Frame: {self.frame_count}",
            f"Tracked Objects: L:{unique_cups_l} R:{unique_cups_r}",
            f"Current detections: L:{len(current_centroids_l)} R:{len(current_centroids_r)}",
            f"FLOW+TAP frames: L:{self.serving_frames_l} R:{self.serving_frames_r} (265f/beer)",
            f"BEERS SERVED: L:{self.beers_served_l} R:{self.beers_served_r}",
            f"Tap Status: L:{'ACTIVE' if tap_l_active else 'IDLE'} R:{'ACTIVE' if tap_r_active else 'IDLE'}",
            f"Total Frames: L:{self.total_smart_servings_l} R:{self.total_smart_servings_r}",
            f"Conf: {self.conf_threshold:.2f} | 'q':quit 'r':reset 't':toggle '+/-':conf"
        ]
        
        for i, text in enumerate(stats_text):
            y_pos = 30 + i * 25
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (10, y_pos - 20), (10 + text_size[0], y_pos + 5), (0, 0, 0), -1)
            cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame, len(current_centroids_l), len(current_centroids_r), smart_servings_l, smart_servings_r

    def run_realtime(self, video_path):
        """Run real-time detection on video"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return
        
        # Video properties
        fps = cap.get(5)  # CAP_PROP_FPS
        width = int(cap.get(3))  # CAP_PROP_FRAME_WIDTH  
        height = int(cap.get(4))  # CAP_PROP_FRAME_HEIGHT
        total_frames = int(cap.get(7))  # CAP_PROP_FRAME_COUNT
        
        print(f"\nVideo: {video_path.name}")
        print(f"Resolution: {width}x{height}, FPS: {fps:.2f}, Total frames: {total_frames}")
        print("\nReal-time detection started...")
        print("Controls:")
        print("- 'q' or 'Q': Quit")
        print("- 'r' or 'R': Reset counters")
        print("- 'SPACE': Pause/Resume")
        print("- '+': Increase confidence")
        print("- '-': Decrease confidence")
        print("- 't' or 'T': Toggle ALL detections / beer-only detections")
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                self.frame_count += 1
                if not ret:
                    print("End of video or playback error")
                    break
                
                # Process frame
                annotated_frame, cups_l, cups_r, servings_l, servings_r = self.process_frame(frame)
                
                # Resize for display if too large
                if width > 1920:
                    scale = 1920 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    annotated_frame = cv2.resize(annotated_frame, (new_width, new_height))
            else:
                # Show paused message
                pause_text = "PAUSED - Press SPACE to resume"
                cv2.putText(annotated_frame, pause_text, (50, height//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            
            # Display frame
            cv2.imshow('Beer Cup Detection - Real Time', annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('r') or key == ord('R'):
                self.total_detections_l = 0
                self.total_detections_r = 0
                self.total_smart_servings_l = 0
                self.total_smart_servings_r = 0
                self.beers_served_l = 0
                self.beers_served_r = 0
                self.serving_frames_l = 0
                self.serving_frames_r = 0
                self.frame_count = 0
                print("All counters reset!")
            elif key == ord(' '):  # Space bar
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == ord('+') or key == ord('='):
                self.conf_threshold = min(0.9, self.conf_threshold + 0.05)
                print(f"Confidence threshold: {self.conf_threshold:.2f}")
            elif key == ord('-'):
                self.conf_threshold = max(0.1, self.conf_threshold - 0.05)
                print(f"Confidence threshold: {self.conf_threshold:.2f}")
            elif key == ord('t') or key == ord('T'):
                self.show_all_detections = not self.show_all_detections
                mode = "ALL detections" if self.show_all_detections else "Beer-only detections"
                print(f"Switched to: {mode}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\nFinal Statistics:")
        print(f"Total frames processed: {self.frame_count}")
        print(f"Total cup detections: L:{self.total_detections_l} R:{self.total_detections_r}")
        print(f"SMART SERVINGS (frames with cup + active tap): L:{self.total_smart_servings_l} R:{self.total_smart_servings_r}")
        print(f"BEERS SERVED (265 frames each, cup in FLOW + active tap): L:{self.beers_served_l} R:{self.beers_served_r}")
        print(f"Frames per beer: {self.min_serving_frames}")
        print(f"Total smart frames: {self.total_smart_servings_l + self.total_smart_servings_r}")
        print(f"Total valid beers served: {self.beers_served_l + self.beers_served_r}")
        print(f"Note: Each 265 consecutive frames (cup in FLOW + active tap) = 1 beer served!")


def main():
    parser = argparse.ArgumentParser(description='Real-time YOLOv8 Cup Detection in Beer Flow ROIs')
    parser.add_argument('video_path', type=str, help='Path to input video')
    parser.add_argument('--model', type=str, default='runs/detect/train_corrected2/weights/best.pt', help='Path to YOLO model (default: fine-tuned model)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.5, help='IOU threshold (default: 0.5)')
    
    args = parser.parse_args()
    
    # Validate paths
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return
    
    # Initialize detector
    detector = RealTimeCupDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Run real-time detection
    detector.run_realtime(video_path)


if __name__ == "__main__":
    main()