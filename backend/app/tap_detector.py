"""
Tap Lever Detection using Template Matching
Detects UP/DOWN position of tap levers using template matching
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from enum import Enum
from dataclasses import dataclass

class TapState(Enum):
    UP = "UP"
    DOWN = "DOWN" 
    UNCERTAIN = "UNCERTAIN"

@dataclass
class TapDetection:
    state: TapState
    confidence_up: float
    confidence_down: float
    frames_stable: int

class TapDetector:
    """Template matching based tap lever detector"""
    
    def __init__(self, templates_path: str = "templates"):
        # Configuration
        self.TEMPLATE_MARGIN = 0.15  # Minimum difference between up/down scores
        self.TAP_DEBOUNCE_FRAMES = 5  # Frames needed for state change
        self.MATCH_THRESHOLD = 0.3   # Minimum template match confidence
        
        # Multi-beer detection within DOWN state
        self.PAUSE_FRAMES = 20       # Frames without flow to consider a pause (1 second at 20fps)
        self.MIN_POUR_FRAMES = 10    # Minimum frames for a valid pour
        self.MAX_SINGLE_BEER_FRAMES = 160  # 8 seconds max for single beer (at 20fps)
        self.INTENSITY_CHANGE_THRESHOLD = 0.3  # Threshold for detecting intensity changes
        
        # Template storage
        self.templates = {}
        self.templates_path = Path(templates_path)
        
        # State tracking
        self.tap_states = {
            'A': TapDetection(TapState.UNCERTAIN, 0.0, 0.0, 0),
            'B': TapDetection(TapState.UNCERTAIN, 0.0, 0.0, 0)
        }
        
        # Debouncing
        self.pending_states = {'A': None, 'B': None}
        self.pending_frames = {'A': 0, 'B': 0}
        
        # Initial state detection flags
        self.initial_detection_done = {'A': False, 'B': False}
        self.was_initially_up = {'A': False, 'B': False}
        
        # Cycle counting - one detection per UP->DOWN cycle
        self.cycle_counted = {'A': False, 'B': False}  # Has this cycle been counted?
        self.waiting_for_up = {'A': False, 'B': False}  # Waiting for UP before next count
        
        # Multi-beer detection within DOWN state
        self.current_pour_active = {'A': False, 'B': False}  # Is currently pouring?
        self.pause_frame_count = {'A': 0, 'B': 0}           # Frames without flow
        self.pour_frame_count = {'A': 0, 'B': 0}            # Frames with flow in current pour
        self.total_pours_in_cycle = {'A': 0, 'B': 0}        # Total pours in this DOWN cycle
        self.current_pour_intensity = {'A': 0.0, 'B': 0.0}  # Current flow intensity
        self.intensity_history = {'A': [], 'B': []}         # Last 20 intensity readings
        self.last_beer_frame = {'A': 0, 'B': 0}            # Frame of last beer detection
        
        # Load templates
        self._load_templates()
    
    def _load_templates(self):
        """Load and preprocess template images"""
        template_files = {
            'A_UP': 'tapA_up.png',
            'A_DOWN': 'tapA_down.png', 
            'B_UP': 'tapB_up.png',
            'B_DOWN': 'tapB_down.png'
        }
        
        for key, filename in template_files.items():
            template_path = self.templates_path / filename
            if template_path.exists():
                # Load template
                template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    # Preprocess: blur to reduce noise
                    template = cv2.GaussianBlur(template, (3, 3), 0)
                    self.templates[key] = template
                    print(f"✅ Loaded template: {filename} ({template.shape})")
                else:
                    print(f"❌ Failed to load template: {filename}")
            else:
                print(f"⚠️ Template not found: {template_path}")
        
        print(f"Loaded {len(self.templates)}/4 templates")
    
    def _preprocess_roi(self, roi: np.ndarray) -> np.ndarray:
        """Preprocess ROI for template matching"""
        if len(roi.shape) == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply gaussian blur to reduce noise
        roi = cv2.GaussianBlur(roi, (3, 3), 0)
        return roi
    
    def _match_templates(self, roi: np.ndarray, tap: str) -> Tuple[float, float]:
        """
        Match templates against ROI - simplified to focus on UP detection
        Returns: (confidence_up, confidence_down)
        """
        roi_processed = self._preprocess_roi(roi)
        
        # Get UP template for this tap (main focus)
        template_up = self.templates.get(f'{tap}_UP')
        
        confidence_up = 0.0
        confidence_down = 0.0  # Not used in new logic but kept for compatibility
        
        if template_up is not None:
            # Match UP template - this is the key detection
            result_up = cv2.matchTemplate(roi_processed, template_up, cv2.TM_CCOEFF_NORMED)
            confidence_up = float(np.max(result_up))
            
            # For DOWN confidence, use inverse logic: if not UP, assume DOWN
            confidence_down = max(0.0, 1.0 - confidence_up)
        
        return confidence_up, confidence_down
    
    def _determine_state(self, confidence_up: float, confidence_down: float) -> TapState:
        """Determine tap state based on template matching scores - simplified for UP detection"""
        # For videos starting with taps already pouring, be more aggressive
        # If UP template doesn't match well, assume DOWN immediately
        if confidence_up > 0.6:  # Strong match with UP template
            return TapState.UP
        else:  # No strong match with UP template - assume DOWN
            return TapState.DOWN
    
    def detect_tap_state(self, roi_tap_a: np.ndarray, roi_tap_b: np.ndarray) -> Dict[str, TapDetection]:
        """
        Detect tap states for both taps with temporal debouncing
        """
        results = {}
        
        for tap, roi in [('A', roi_tap_a), ('B', roi_tap_b)]:
            if roi is None:
                results[tap] = self.tap_states[tap]
                continue
                
            # Get template matching scores
            conf_up, conf_down = self._match_templates(roi, tap)
            
            # Determine current state
            current_state = self._determine_state(conf_up, conf_down)
            
            # Apply temporal debouncing
            if not self.initial_detection_done[tap]:
                # First detection - set state immediately and remember initial state
                self.tap_states[tap].state = current_state
                self.tap_states[tap].frames_stable = 1
                self.initial_detection_done[tap] = True
                self.was_initially_up[tap] = (current_state == TapState.UP)
                
                # If started DOWN, mark for immediate counting
                if current_state == TapState.DOWN:
                    self.cycle_counted[tap] = False  # Allow first count
                    self.waiting_for_up[tap] = True   # Need UP before next cycle
                
                print(f"🔄 Tap {tap} initial state: {current_state.value} (initially_up: {self.was_initially_up[tap]})")
            elif current_state == self.tap_states[tap].state:
                # State is stable, increase stability counter
                self.tap_states[tap].frames_stable += 1
                self.pending_states[tap] = None
                self.pending_frames[tap] = 0
            else:
                # State change detected
                if self.pending_states[tap] == current_state:
                    # Same pending state, increment counter
                    self.pending_frames[tap] += 1
                    
                    # Check if we have enough stable frames
                    if self.pending_frames[tap] >= self.TAP_DEBOUNCE_FRAMES:
                        # Accept state change
                        old_state = self.tap_states[tap].state
                        self.tap_states[tap].state = current_state
                        self.tap_states[tap].frames_stable = self.pending_frames[tap]
                        self.pending_states[tap] = None
                        self.pending_frames[tap] = 0
                        
                        # Handle cycle counting logic
                        if old_state == TapState.UP and current_state == TapState.DOWN:
                            # UP -> DOWN transition: allow new count
                            self.cycle_counted[tap] = False
                            self.waiting_for_up[tap] = True
                            print(f"🔄 Tap {tap} UP->DOWN: New cycle ready for counting")
                        elif old_state == TapState.DOWN and current_state == TapState.UP:
                            # DOWN -> UP transition: reset for next cycle
                            self.waiting_for_up[tap] = False
                            self.cycle_counted[tap] = False
                            self.total_pours_in_cycle[tap] = 0
                            self.current_pour_active[tap] = False
                            self.pause_frame_count[tap] = 0
                            self.pour_frame_count[tap] = 0
                            print(f"🔄 Tap {tap} DOWN->UP: Cycle reset ({self.total_pours_in_cycle[tap]} pours detected)")
                        else:
                            print(f"🔄 Tap {tap} state changed to {current_state.value}")
                else:
                    # Different pending state, reset counter
                    self.pending_states[tap] = current_state
                    self.pending_frames[tap] = 1
            
            # Update confidence scores
            self.tap_states[tap].confidence_up = conf_up
            self.tap_states[tap].confidence_down = conf_down
            
            results[tap] = self.tap_states[tap]
        
        return results
    
    def should_allow_pour(self, tap: str, flow_active: bool, flow_intensity: float = 0.0) -> bool:
        """
        Detect multiple beers using duration and intensity analysis.
        
        Estrategias para cervezas continuas:
        1. División por duración (>8s = múltiples cervezas)
        2. Cambios en intensidad del flujo
        3. Pausas en el flujo
        """
        tap_detection = self.tap_states.get(tap)
        if tap_detection is None:
            return False
        
        # Check if we have UP template for this tap
        up_template_key = f'{tap}_UP'
        if up_template_key not in self.templates:
            return flow_active  # Fallback to flow-only
        
        # Must be DOWN to pour
        if tap_detection.state != TapState.DOWN:
            # Not DOWN - reset pour tracking
            self._reset_pour_tracking(tap)
            return False
        
        # Tap is DOWN - analyze flow patterns
        if flow_active:
            return self._analyze_continuous_flow(tap, flow_intensity)
        else:
            return self._analyze_flow_pause(tap)
    
    def _reset_pour_tracking(self, tap: str):
        """Reset all pour tracking variables for a tap"""
        self.current_pour_active[tap] = False
        self.pause_frame_count[tap] = 0
        self.pour_frame_count[tap] = 0
        self.intensity_history[tap] = []
        self.current_pour_intensity[tap] = 0.0
    
    def _analyze_continuous_flow(self, tap: str, flow_intensity: float) -> bool:
        """Analyze continuous flow for multiple beer detection"""
        # Flow detected - reset pause counter
        self.pause_frame_count[tap] = 0
        self.pour_frame_count[tap] += 1
        
        # Update intensity tracking
        self.current_pour_intensity[tap] = flow_intensity
        self.intensity_history[tap].append(flow_intensity)
        if len(self.intensity_history[tap]) > 20:
            self.intensity_history[tap].pop(0)  # Keep only last 20 readings
        
        if not self.current_pour_active[tap]:
            # Starting new pour
            if self.pour_frame_count[tap] >= self.MIN_POUR_FRAMES:
                self.current_pour_active[tap] = True
                self.total_pours_in_cycle[tap] += 1
                self.last_beer_frame[tap] = self.pour_frame_count[tap]
                print(f"🍺 Tap {tap}: Pour #{self.total_pours_in_cycle[tap]} started")
                return True
        else:
            # Continue existing pour - check for multiple beer indicators
            frames_since_last = self.pour_frame_count[tap] - self.last_beer_frame[tap]
            
            # Strategy 1: Duration-based detection
            if frames_since_last >= self.MAX_SINGLE_BEER_FRAMES:
                self.total_pours_in_cycle[tap] += 1
                self.last_beer_frame[tap] = self.pour_frame_count[tap]
                print(f"🍺 Tap {tap}: Pour #{self.total_pours_in_cycle[tap]} (duration-based, {frames_since_last/20:.1f}s)")
                return True
            
            # Strategy 2: Intensity change detection
            if len(self.intensity_history[tap]) >= 10:
                recent_avg = sum(self.intensity_history[tap][-5:]) / 5
                older_avg = sum(self.intensity_history[tap][-10:-5]) / 5
                
                if older_avg > 0 and abs(recent_avg - older_avg) / older_avg > self.INTENSITY_CHANGE_THRESHOLD:
                    # Significant intensity change detected
                    if frames_since_last >= self.MIN_POUR_FRAMES * 2:  # At least 1 second since last
                        self.total_pours_in_cycle[tap] += 1
                        self.last_beer_frame[tap] = self.pour_frame_count[tap]
                        print(f"🍺 Tap {tap}: Pour #{self.total_pours_in_cycle[tap]} (intensity change: {older_avg:.3f}→{recent_avg:.3f})")
                        return True
        
        return False
    
    def _analyze_flow_pause(self, tap: str) -> bool:
        """Analyze flow pause for beer detection"""
        if self.current_pour_active[tap]:
            self.pause_frame_count[tap] += 1
            
            # Check if pause is long enough to end current pour
            if self.pause_frame_count[tap] >= self.PAUSE_FRAMES:
                print(f"⏸️ Tap {tap}: Pour #{self.total_pours_in_cycle[tap]} ended (pause detected)")
                self.current_pour_active[tap] = False
                self.pour_frame_count[tap] = 0  # Reset for next pour
        
        return False
    
    def get_debug_info(self, tap: str) -> str:
        """Get debug string for tap state"""
        detection = self.tap_states.get(tap)
        if detection is None:
            return f"Tap {tap}: NO DATA"
        
        state_str = detection.state.value
        up_score = detection.confidence_up
        down_score = detection.confidence_down
        stable_frames = detection.frames_stable
        
        return f"Tap {tap}: {state_str} ({up_score:.2f}/{down_score:.2f}) [{stable_frames}f]"
    
    def has_templates(self) -> bool:
        """Check if templates are loaded"""
        return len(self.templates) >= 2  # At least some templates loaded
    
    def get_required_templates(self) -> List[str]:
        """Get list of required template files"""
        return ['tapA_up.png', 'tapA_down.png', 'tapB_up.png', 'tapB_down.png']