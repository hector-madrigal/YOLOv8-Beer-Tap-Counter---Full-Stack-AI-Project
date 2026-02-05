"""
Video processing service for beer pour detection.

This module implements computer vision algorithms to detect and count
beer pours from a dual-tap system using motion detection, region analysis,
and template matching for tap lever detection.
"""
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import re

from app.config import FRAME_SKIP, MOTION_THRESHOLD, MIN_POUR_DURATION_FRAMES
from app.roi_config import get_roi, extract_roi
from app.tap_detector import TapDetector, TapState

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


@dataclass
class PourDetection:
    """Represents a detected beer pour event."""
    tap: str  # 'A' or 'B'
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    confidence: float = 1.0


@dataclass
class VideoAnalysisResult:
    """Complete video analysis result including timestamp."""
    detections: List[PourDetection]
    duration: float
    video_timestamp: Optional[str] = None


class BeerPourDetector:
    """
    Detects beer pours from video using motion detection and region analysis.
    
    The detector divides the frame into left (Tap A) and right (Tap B) regions
    and tracks motion patterns that indicate a beer being poured.
    """
    
    def __init__(
        self,
        frame_skip: int = FRAME_SKIP,
        motion_threshold: int = MOTION_THRESHOLD,
        min_pour_duration: int = MIN_POUR_DURATION_FRAMES
    ):
        self.frame_skip = frame_skip
        self.motion_threshold = motion_threshold
        self.min_pour_duration = min_pour_duration
        
        # State tracking
        self.tap_a_active = False
        self.tap_b_active = False
        self.tap_a_start_frame = 0
        self.tap_b_start_frame = 0
        
        # Motion history
        self.motion_history_a: List[float] = []
        self.motion_history_b: List[float] = []
        self.history_size = 10
        
        # Detected pours
        self.detections: List[PourDetection] = []
        
        # Background subtractors for motion detection (separate for each flow ROI)
        self.bg_subtractor_a = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=False
        )
        self.bg_subtractor_b = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=False
        )
        
        # Template matching tap detector
        self.tap_detector = TapDetector()
        print(f"🎯 Tap detector initialized. Templates loaded: {self.tap_detector.has_templates()}")
        if not self.tap_detector.has_templates():
            print("⚠️ No templates found. Tap detection disabled.")
            required = self.tap_detector.get_required_templates()
            print(f"Required templates: {required}")
        
        # First pour flag removed - now allows pours immediately if tap not UP
        
    def _get_tap_regions(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract the specific ROI regions for flow detection using configured ROIs.
        
        Returns the flow regions (where beer pours) for both taps.
        """
        # Extract flow regions using the manually configured ROIs
        tap_a_region = extract_roi(frame, 'ROI_FLOW_L')  # Left flow region
        tap_b_region = extract_roi(frame, 'ROI_FLOW_R')  # Right flow region
        
        # NO FALLBACK - must use configured ROIs only
        if tap_a_region is None:
            print(f"❌ ERROR: Could not extract ROI_FLOW_L from frame")
            
        if tap_b_region is None:
            print(f"❌ ERROR: Could not extract ROI_FLOW_R from frame")
            
        return tap_a_region, tap_b_region
        
    def _get_tap_lever_regions(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract the tap lever ROI regions for template matching.
        
        Returns the tap lever regions for both taps.
        """
        # Extract tap lever regions using the manually configured ROIs
        tap_a_lever = extract_roi(frame, 'ROI_TAP_L')  # Left tap lever region
        tap_b_lever = extract_roi(frame, 'ROI_TAP_R')  # Right tap lever region
        
        # NO FALLBACK - must use configured ROIs only
        if tap_a_lever is None:
            print(f"❌ ERROR: Could not extract ROI_TAP_L from frame")
            
        if tap_b_lever is None:
            print(f"❌ ERROR: Could not extract ROI_TAP_R from frame")
            
        return tap_a_lever, tap_b_lever
        
    def _get_tap_info_region(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract the timestamp region for temporal information.
        """
        return extract_roi(frame, 'ROI_TS')
    
    def _extract_timestamp_from_frame(self, frame: np.ndarray) -> Optional[str]:
        """
        Extract timestamp text from the timestamp ROI using OCR.
        
        Returns the extracted timestamp string or None if extraction fails.
        """
        if not TESSERACT_AVAILABLE:
            return None
            
        # Extract timestamp region
        ts_region = self._get_tap_info_region(frame)
        if ts_region is None:
            return None
        
        try:
            # Preprocess for better OCR
            gray = cv2.cvtColor(ts_region, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to get better contrast
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Use pytesseract to extract text
            text = pytesseract.image_to_string(thresh, config='--psm 7')
            
            # Clean up the text
            text = text.strip()
            
            # Basic validation - check if it looks like a timestamp
            # Common formats: "2024-01-15 14:30:45", "15/01/2024 14:30:45", etc.
            if len(text) > 10 and any(char.isdigit() for char in text):
                return text
                
        except Exception as e:
            print(f"Error extracting timestamp: {e}")
            
        return None
    
    def _calculate_motion(self, region: np.ndarray, is_tap_a: bool = True) -> float:
        """
        Calculate motion intensity in a region using background subtraction.
        
        Returns a normalized value between 0 and 1 indicating motion level.
        """
        # Use the appropriate background subtractor for each tap
        bg_subtractor = self.bg_subtractor_a if is_tap_a else self.bg_subtractor_b
        
        # Apply background subtraction
        fg_mask = bg_subtractor.apply(region)
        
        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Calculate motion percentage
        motion_pixels = np.sum(fg_mask > 0)
        total_pixels = fg_mask.size
        motion_ratio = motion_pixels / total_pixels if total_pixels > 0 else 0
        
        return motion_ratio
    
    def _detect_pour_motion(self, region: np.ndarray, is_tap_a: bool = True) -> Tuple[bool, float]:
        """
        Detect if there's pouring motion in the region.
        
        A pour is characterized by:
        1. Sustained motion in the tap area
        2. Vertical movement patterns (liquid flowing down)
        
        Returns: (is_pouring, confidence)
        """
        # Convert to grayscale
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region
            
        # Calculate motion using frame differencing
        motion_ratio = self._calculate_motion(region, is_tap_a)
        
        # Update motion history
        history = self.motion_history_a if is_tap_a else self.motion_history_b
        history.append(motion_ratio)
        if len(history) > self.history_size:
            history.pop(0)
        
        # Check for sustained motion pattern
        if len(history) >= 3:
            avg_motion = np.mean(history[-3:])
            is_active = avg_motion > 0.02  # Reasonable threshold for small ROIs
            
            # Calculate confidence based on motion consistency
            if is_active and len(history) >= 5:
                motion_std = np.std(history[-5:])
                confidence = min(1.0, avg_motion * 10) * (1 - min(motion_std, 0.5))
            else:
                confidence = avg_motion * 10
                
            return is_active, min(1.0, confidence)
        
        return False, 0.0
    
    def _should_start_pour(self, tap: str, flow_active: bool, flow_intensity: float = 0.0) -> bool:
        """
        Determine if a pour should start based on flow and tap state.
        Integrates template matching with motion detection.
        """
        if not flow_active:
            return False
        
        # If no templates loaded, use flow-only detection
        if not self.tap_detector.has_templates():
            return True
        
        # Use template matching to validate pour with intensity
        should_allow = self.tap_detector.should_allow_pour(tap, flow_active, flow_intensity)
        
        return should_allow
    
    def _analyze_frame_optical_flow(self, prev_gray: np.ndarray, curr_gray: np.ndarray, 
                                      is_left: bool) -> Tuple[bool, float]:
        """
        Use optical flow to detect downward motion characteristic of pouring.
        """
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Get vertical component (y-direction)
        flow_y = flow[..., 1]
        
        # Pouring creates downward flow (positive y values)
        downward_flow = np.sum(flow_y > 2)  # Threshold for significant downward motion
        total_pixels = flow_y.size
        
        downward_ratio = downward_flow / total_pixels if total_pixels > 0 else 0
        
        # Detect if there's significant downward motion
        is_pouring = downward_ratio > 0.02
        confidence = min(1.0, downward_ratio * 20)
        
        return is_pouring, confidence
    
    def process_video(self, video_path: str) -> VideoAnalysisResult:
        """
        Process a video file and detect all beer pour events.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            VideoAnalysisResult with detections, duration, and extracted timestamp
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Reset state
        self.detections = []
        self.tap_a_active = False
        self.tap_b_active = False
        self.motion_history_a = []
        self.motion_history_b = []
        
        # Log ROI information
        roi_flow_l = get_roi('ROI_FLOW_L')
        roi_flow_r = get_roi('ROI_FLOW_R')
        roi_ts = get_roi('ROI_TS')
        print(f"Using ROIs:")
        print(f"  Flow Left (Tap A): {roi_flow_l}")
        print(f"  Flow Right (Tap B): {roi_flow_r}")
        print(f"  Timestamp: {roi_ts}")
        
        # Extract timestamp from first frame
        video_timestamp = None
        ret_first, first_frame = cap.read()
        if ret_first:
            video_timestamp = self._extract_timestamp_from_frame(first_frame)
            if video_timestamp:
                print(f"  Extracted timestamp: {video_timestamp}")
            # Reset video to start
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Reset background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=False
        )
        
        prev_frame = None
        frame_count = 0
        
        # Tracking variables for each tap
        tap_a_motion_frames = 0
        tap_b_motion_frames = 0
        tap_a_no_motion_frames = 0
        tap_b_no_motion_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for performance
            if frame_count % self.frame_skip != 0:
                frame_count += 1
                continue
            
            # Get tap regions
            tap_a_region, tap_b_region = self._get_tap_regions(frame)
            
            # Convert to grayscale for analysis
            tap_a_gray = cv2.cvtColor(tap_a_region, cv2.COLOR_BGR2GRAY)
            tap_b_gray = cv2.cvtColor(tap_b_region, cv2.COLOR_BGR2GRAY)
            
            # Detect motion in each region
            is_pouring_a, conf_a = self._detect_pour_motion(tap_a_region, is_tap_a=True)
            is_pouring_b, conf_b = self._detect_pour_motion(tap_b_region, is_tap_a=False)
            
            # State machine for Tap A
            if is_pouring_a:
                tap_a_motion_frames += 1
                tap_a_no_motion_frames = 0
                
                if not self.tap_a_active and tap_a_motion_frames >= 3:
                    # Start of pour detected
                    self.tap_a_active = True
                    self.tap_a_start_frame = frame_count - (3 * self.frame_skip)
            else:
                tap_a_no_motion_frames += 1
                
                if self.tap_a_active and tap_a_no_motion_frames >= 3:
                    # End of pour detected
                    pour_duration = frame_count - self.tap_a_start_frame
                    
                    if pour_duration >= self.min_pour_duration:
                        detection = PourDetection(
                            tap='A',
                            start_frame=self.tap_a_start_frame,
                            end_frame=frame_count,
                            start_time=self.tap_a_start_frame / fps,
                            end_time=frame_count / fps,
                            confidence=conf_a
                        )
                        self.detections.append(detection)
                    
                    self.tap_a_active = False
                    tap_a_motion_frames = 0
            
            # State machine for Tap B
            if is_pouring_b:
                tap_b_motion_frames += 1
                tap_b_no_motion_frames = 0
                
                if not self.tap_b_active and tap_b_motion_frames >= 3:
                    # Start of pour detected
                    self.tap_b_active = True
                    self.tap_b_start_frame = frame_count - (3 * self.frame_skip)
            else:
                tap_b_no_motion_frames += 1
                
                if self.tap_b_active and tap_b_no_motion_frames >= 3:
                    # End of pour detected
                    pour_duration = frame_count - self.tap_b_start_frame
                    
                    if pour_duration >= self.min_pour_duration:
                        detection = PourDetection(
                            tap='B',
                            start_frame=self.tap_b_start_frame,
                            end_frame=frame_count,
                            start_time=self.tap_b_start_frame / fps,
                            end_time=frame_count / fps,
                            confidence=conf_b
                        )
                        self.detections.append(detection)
                    
                    self.tap_b_active = False
                    tap_b_motion_frames = 0
            
            frame_count += 1
        
        # Handle any ongoing pours at video end
        if self.tap_a_active:
            pour_duration = frame_count - self.tap_a_start_frame
            if pour_duration >= self.min_pour_duration:
                detection = PourDetection(
                    tap='A',
                    start_frame=self.tap_a_start_frame,
                    end_frame=frame_count,
                    start_time=self.tap_a_start_frame / fps,
                    end_time=frame_count / fps,
                    confidence=0.8
                )
                self.detections.append(detection)
        
        if self.tap_b_active:
            pour_duration = frame_count - self.tap_b_start_frame
            if pour_duration >= self.min_pour_duration:
                detection = PourDetection(
                    tap='B',
                    start_frame=self.tap_b_start_frame,
                    end_frame=frame_count,
                    start_time=self.tap_b_start_frame / fps,
                    end_time=frame_count / fps,
                    confidence=0.8
                )
                self.detections.append(detection)
        
        cap.release()
        
        # Post-process detections to merge close events and filter false positives
        self.detections = self._post_process_detections(self.detections, fps)
        
        return VideoAnalysisResult(
            detections=self.detections,
            duration=duration,
            video_timestamp=video_timestamp
        )
    
    def _post_process_detections(self, detections: List[PourDetection], fps: float) -> List[PourDetection]:
        """
        Post-process detections to merge close events and filter false positives.
        """
        if not detections:
            return []
        
        # Sort by start time
        detections.sort(key=lambda x: (x.tap, x.start_frame))
        
        # Merge close events for each tap
        merged = []
        min_gap_frames = int(fps * 1.0)  # Minimum 1 second gap between separate pours
        
        for tap in ['A', 'B']:
            tap_detections = [d for d in detections if d.tap == tap]
            
            if not tap_detections:
                continue
            
            current = tap_detections[0]
            
            for next_det in tap_detections[1:]:
                gap = next_det.start_frame - current.end_frame
                
                if gap < min_gap_frames:
                    # Merge detections
                    current = PourDetection(
                        tap=current.tap,
                        start_frame=current.start_frame,
                        end_frame=next_det.end_frame,
                        start_time=current.start_time,
                        end_time=next_det.end_time,
                        confidence=max(current.confidence, next_det.confidence)
                    )
                else:
                    merged.append(current)
                    current = next_det
            
            merged.append(current)
        
        # Sort final list by start time
        merged.sort(key=lambda x: x.start_frame)
        
        return merged


def process_video_file(video_path: str) -> VideoAnalysisResult:
    """
    Convenience function to process a video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        VideoAnalysisResult with detections, duration, and timestamp
    """
    detector = BeerPourDetector()
    return detector.process_video(video_path)
