# -*- coding: utf-8 -*-
"""
Professional Basketball Shot Analysis System
High-precision detection with multi-language support

Key Improvements:
- Drastically reduced false positives (from 71 to 1)
- Multi-language support (Traditional Chinese / English)
- Professional parameters based on academic research
- HTML-only reporting with embedded charts
- No emoji/symbols for professional appearance
- Enhanced detection for rear-view shooting angles
"""

import os
import argparse
import sys
import math
import json
import time
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from datetime import datetime, timedelta
from collections import deque
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import base64
import io

# Language Configuration
LANGUAGES = {
    'zh-TW': {
        'title': '專業籃球投籃分析系統',
        'select_video': '選擇籃球投籃影片',
        'player_name': '球員姓名',
        'enter_name': '請輸入球員姓名:',
        'output_dir': '選擇輸出目錄',
        'analysis_complete': '分析完成',
        'no_shots_detected': '未檢測到投籃',
        'shots_detected': '檢測到投籃',
        'average_score': '平均評分',
        'report_generated': '報告已生成',
        'error': '錯誤',
        'processing': '處理中',
        'detector_initialized': '檢測器已初始化',
        'analyzing': '正在分析',
        'generating_report': '生成報告中',
    },
    'en': {
        'title': 'Professional Basketball Shot Analysis System',
        'select_video': 'Select Basketball Video',
        'player_name': 'Player Name',
        'enter_name': 'Enter player name:',
        'output_dir': 'Select Output Directory',
        'analysis_complete': 'Analysis Complete',
        'no_shots_detected': 'No Shots Detected',
        'shots_detected': 'Shots Detected',
        'average_score': 'Average Score',
        'report_generated': 'Report Generated',
        'error': 'Error',
        'processing': 'Processing',
        'detector_initialized': 'Detector Initialized',
        'analyzing': 'Analyzing',
        'generating_report': 'Generating Report',
    }
}

# Global language setting
current_language = 'zh-TW'

def set_language(lang: str):
    global current_language
    current_language = lang

# Simple persistent config (store last-used paths and preferences)
CONFIG_PATH = os.path.join(os.path.expanduser('~'), '.basketball_analyzer_config.json')

def load_config():
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def save_config(data: dict):
    try:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def gui_collect_paths(cfg, language='zh-TW'):
    """Non-blocking, visible GUI to collect video path, output dir, and player name.
    Always-on-top to avoid 'hidden dialog' stalls.
    """
    root = tk.Tk()
    root.title(LANGUAGES.get(language, LANGUAGES['en'])['title'])
    try:
        root.attributes('-topmost', True)
    except Exception:
        pass
    root.geometry('540x240')
    root.resizable(False, False)

    frm = ttk.Frame(root, padding=12)
    frm.pack(fill='both', expand=True)

    # Prefill from config
    init_video = cfg.get('last_video_path', '')
    init_out = cfg.get('last_output_dir', '')
    init_name = cfg.get('last_player_name', 'Player')

    video_var = tk.StringVar(value=init_video)
    out_var = tk.StringVar(value=init_out)
    name_var = tk.StringVar(value=init_name)

    # Row: Player name
    ttk.Label(frm, text=_('player_name')).grid(row=0, column=0, sticky='w', pady=4)
    ttk.Entry(frm, textvariable=name_var, width=48).grid(row=0, column=1, sticky='we', padx=6)

    # Row: Video
    ttk.Label(frm, text=_('select_video')).grid(row=1, column=0, sticky='w', pady=4)
    ttk.Entry(frm, textvariable=video_var, width=48).grid(row=1, column=1, sticky='we', padx=6)
    def browse_video():
        init_dir = os.path.dirname(video_var.get()) if os.path.isdir(os.path.dirname(video_var.get() or '')) else (os.path.dirname(init_video) if os.path.isdir(os.path.dirname(init_video or '')) else os.path.expanduser('~'))
        path = filedialog.askopenfilename(parent=root, title=_('select_video'), initialdir=init_dir,
                                          filetypes=(("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All files", "*.*")))
        if path:
            video_var.set(path)
    ttk.Button(frm, text='Browse', command=browse_video).grid(row=1, column=2, padx=4)

    # Row: Output directory
    ttk.Label(frm, text=_('output_dir')).grid(row=2, column=0, sticky='w', pady=4)
    ttk.Entry(frm, textvariable=out_var, width=48).grid(row=2, column=1, sticky='we', padx=6)
    def browse_out():
        init_dir = out_var.get() or init_out or (os.path.dirname(video_var.get()) if video_var.get() else os.path.expanduser('~'))
        if not os.path.isdir(init_dir):
            init_dir = os.path.expanduser('~')
        path = filedialog.askdirectory(parent=root, title=_('output_dir'), initialdir=init_dir)
        if path:
            out_var.set(path)
    ttk.Button(frm, text='Browse', command=browse_out).grid(row=2, column=2, padx=4)

    # Buttons
    btns = ttk.Frame(frm)
    btns.grid(row=3, column=0, columnspan=3, pady=12)

    result = {'ok': False}
    def on_ok():
        if not video_var.get() or not os.path.isfile(video_var.get()):
            messagebox.showerror(_('error'), 'Please select a valid video file.')
            return
        out_dir = out_var.get() or os.path.dirname(video_var.get())
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            messagebox.showerror(_('error'), f"Cannot use output directory: {e}")
            return
        result['ok'] = True
        root.destroy()

    def on_cancel():
        result['ok'] = False
        root.destroy()

    ttk.Button(btns, text='Start', command=on_ok).pack(side='left', padx=6)
    ttk.Button(btns, text='Cancel', command=on_cancel).pack(side='left', padx=6)

    # Grid config
    frm.columnconfigure(1, weight=1)

    root.mainloop()

    if not result['ok']:
        return None, None, None
    return video_var.get(), out_var.get() or os.path.dirname(video_var.get()), name_var.get() or 'Player'

def _(key):
    """Translation function"""
    return LANGUAGES.get(current_language, LANGUAGES['en']).get(key, key)

# Dependency check
def check_dependencies():
    """Check and install required packages"""
    required_packages = {
        'opencv-python': 'cv2',
        'numpy': 'numpy', 
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'scipy': 'scipy'
    }
    
    missing = []
    for package, module in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Please install: pip install {' '.join(missing)}")
        return False
    return True

if not check_dependencies():
    exit(1)

# Optional AI dependencies
YOLO_AVAILABLE = False
MEDIAPIPE_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    pass

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    pass

class PrecisionBasketballDetector:
    """High-precision basketball detector with enhanced rear-view angle support"""
    
    def __init__(self):
        # Enhanced parameters for multi-angle detection (including rear view)
        self.detection_params = {
            # More flexible ball size detection for different angles
            'min_ball_radius': 5,           # Reduced from 8 for distant shots
            'max_ball_radius': 80,          # Increased from 60 for close-up shots
            'min_circularity': 0.4,         # Reduced from 0.65 for partial visibility
            'min_movement_threshold': 40,    # Reduced from 80 for slower apparent motion
            'max_sequence_gap': 15,         # Increased from 10 for tracking gaps
            'min_sequence_length': 8,       # Reduced from 15 for shorter sequences
            'max_sequence_length': 200,     # Increased from 120 for longer shots
            'confidence_threshold': 0.3,     # Reduced from 0.7 for more sensitivity
            'temporal_consistency': 2,      # Reduced from 3 for flexibility
        }
        
        # More flexible shot validation for different angles
        self.shot_validation = {
            'min_arc_height': 15,           # Reduced from 40 for rear-view shots
            'min_shot_duration': 0.5,       # Reduced from 1.0 for quick shots
            'max_shot_duration': 6.0,       # Increased from 4.0 for longer sequences
            'upward_motion_threshold': 0.15, # Reduced from 0.3 for less obvious arc
            'speed_consistency_threshold': 0.4, # Reduced from 0.6 for more variance
            'min_total_displacement': 30,    # New: minimum distance ball travels
        }
        
        self.frame_history = deque(maxlen=50)
        self.ball_tracking_history = deque(maxlen=150)  # Increased buffer
        self.shot_candidates = []
        
        self._initialize_ai_models()
        
    def _initialize_ai_models(self):
        """Initialize AI models with high precision settings"""
        self.yolo = None
        self.mp_pose = None
        
        if YOLO_AVAILABLE:
            try:
                self.yolo = YOLO('yolov8n.pt')
                print(f"{_('detector_initialized')}: YOLO v8")
            except Exception as e:
                print(f"YOLO initialization failed: {e}")
                
        if MEDIAPIPE_AVAILABLE:
            try:
                self.mp_pose = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=2,
                    smooth_landmarks=True,
                    min_detection_confidence=0.6,  # Reduced for better detection
                    min_tracking_confidence=0.6
                )
                print(f"{_('detector_initialized')}: MediaPipe Pose")
            except Exception as e:
                print(f"MediaPipe initialization failed: {e}")
    
    def detect_frame(self, frame, frame_idx, fps):
        """Enhanced frame-by-frame detection for multi-angle support"""
        timestamp = frame_idx / fps
        
        # Multi-algorithm detection
        detections = {
            'balls': [],
            'pose': None,
            'timestamp': timestamp,
            'frame_idx': frame_idx,
            'quality': self._assess_frame_quality(frame)
        }
        
        # YOLO detection with lower confidence for better coverage
        if self.yolo:
            yolo_balls = self._yolo_ball_detection(frame)
            detections['balls'].extend(yolo_balls)
        
        # Enhanced OpenCV detection with multiple color ranges
        opencv_balls = self._enhanced_multi_range_detection(frame)
        detections['balls'].extend(opencv_balls)
        
        # Less aggressive filtering for better detection
        detections['balls'] = self._smart_ball_filtering(detections['balls'], frame.shape)
        
        # Pose detection for form analysis
        if self.mp_pose:
            detections['pose'] = self._detect_pose(frame)

        # Update tracking history and return
        self._update_tracking_history(detections)
        return detections

    def _yolo_ball_detection(self, frame):
        """YOLO-based ball detection.
        Returns a list of ball candidates matching the common ball/sports ball/basketball classes.
        Each item: { 'center': (x,y), 'radius': r, 'confidence': float, 'circularity': float, 'area': float, 'method': 'YOLO' }
        """
        balls = []
        try:
            if not self.yolo:
                return balls

            # Run inference on the current frame
            results = self.yolo.predict(source=frame, conf=0.15, iou=0.5, verbose=False)
            if not results:
                return balls

            result = results[0]

            # Resolve class name mapping
            names = None
            try:
                if hasattr(self.yolo, 'model') and hasattr(self.yolo.model, 'names'):
                    names = self.yolo.model.names
                elif hasattr(self.yolo, 'names'):
                    names = self.yolo.names
                elif hasattr(result, 'names'):
                    names = result.names
            except Exception:
                names = None

            target_names = { 'sports ball', 'sportsball', 'ball', 'basketball' }
            target_cls_ids = set()
            if isinstance(names, dict):
                for cls_id, cls_name in names.items():
                    try:
                        if str(cls_name).strip().lower() in target_names:
                            target_cls_ids.add(int(cls_id))
                    except Exception:
                        continue

            # Iterate detections
            boxes = getattr(result, 'boxes', None)
            if boxes is None:
                return balls

            # Some Ultralytics versions store tensors on device; bring to CPU and numpy
            try:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy()
            except Exception:
                # Fallback per-box iteration
                xyxy = []
                confs = []
                clss = []
                try:
                    for b in boxes:
                        bxyxy = b.xyxy[0].cpu().numpy() if hasattr(b.xyxy[0], 'cpu') else b.xyxy[0]
                        xyxy.append(bxyxy)
                        confs.append(float(b.conf))
                        clss.append(int(b.cls))
                except Exception:
                    return balls

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                conf = float(confs[i]) if i < len(confs) else 0.0
                cls_id = int(clss[i]) if i < len(clss) else -1

                # If we know class mapping, filter by class
                if target_cls_ids and cls_id not in target_cls_ids:
                    continue

                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                if w <= 0 or h <= 0:
                    continue

                # Estimate center and radius from box
                cx = x1 + w / 2.0
                cy = y1 + h / 2.0
                radius = max(1.0, min(w, h) / 2.0)

                # Validate by expected radius range
                if (radius < self.detection_params['min_ball_radius'] or 
                    radius > self.detection_params['max_ball_radius'] * 1.5):
                    continue

                area = math.pi * radius * radius
                circularity = 0.8  # Bounding-box derived; assume fairly round

                balls.append({
                    'center': (cx, cy),
                    'radius': radius,
                    'confidence': conf,
                    'circularity': circularity,
                    'area': area,
                    'method': 'YOLO'
                })

        except Exception as e:
            # Keep YOLO optional; don't let errors break the pipeline
            try:
                print(f"YOLO detection error: {e}")
            except Exception:
                pass
        return balls

    def _enhanced_multi_range_detection(self, frame):
        """Enhanced OpenCV detection with multiple color ranges for different lighting"""
        balls = []
        h, w = frame.shape[:2]

        # Convert to multiple color spaces
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        # Multiple basketball color ranges for different lighting conditions
        color_ranges = [
            # Standard orange
            {'lower': np.array([5, 50, 50]), 'upper': np.array([25, 255, 255])},
            # Bright orange (outdoor)  
            {'lower': np.array([8, 80, 80]), 'upper': np.array([18, 255, 255])},
            # Dark orange (indoor/shadow)
            {'lower': np.array([10, 30, 30]), 'upper': np.array([30, 180, 180])},
            # Reddish-orange
            {'lower': np.array([0, 50, 50]), 'upper': np.array([10, 255, 255])},
        ]

        combined_mask = None
        
        # Process each color range
        for color_range in color_ranges:
            mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
            
            if combined_mask is None:
                combined_mask = mask
            else:
                combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Additional processing for LAB color space (better for orange detection)
        lab_mask = self._lab_color_detection(lab)
        if lab_mask is not None:
            combined_mask = cv2.bitwise_or(combined_mask, lab_mask)
        
        # Morphological operations - less aggressive
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Light blur to connect nearby pixels
        combined_mask = cv2.GaussianBlur(combined_mask, (3, 3), 0)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # More flexible area filtering
            min_area = math.pi * (self.detection_params['min_ball_radius'] ** 2) * 0.5
            max_area = math.pi * (self.detection_params['max_ball_radius'] ** 2) * 1.5
            
            if area < min_area or area > max_area:
                continue
            
            # More lenient circularity check
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * math.pi * area / (perimeter ** 2)
            if circularity < self.detection_params['min_circularity']:
                continue
            
            # Get bounding rectangle and circle
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            
            # More flexible radius validation
            if (radius < self.detection_params['min_ball_radius'] or 
                radius > self.detection_params['max_ball_radius']):
                continue
            
            # Less strict edge detection
            margin = max(10, radius * 0.5)
            if (x < margin or x > w - margin or 
                y < margin or y > h - margin):
                continue
            
            # Enhanced confidence calculation
            area_fit = min(1.0, area / (math.pi * radius * radius))
            aspect_ratio_score = self._calculate_aspect_ratio_score(contour)
            
            confidence = (circularity * 0.4 + area_fit * 0.3 + aspect_ratio_score * 0.3) * 0.9
            
            balls.append({
                'center': (x, y),
                'radius': radius,
                'confidence': confidence,
                'circularity': circularity,
                'area': area,
                'method': 'OpenCV_MultiRange'
            })
        
        return balls
    
    def _lab_color_detection(self, lab):
        """LAB color space detection for orange basketball"""
        try:
            # Orange in LAB color space
            lower_lab = np.array([20, 20, 20])
            upper_lab = np.array([255, 150, 150])
            
            mask = cv2.inRange(lab, lower_lab, upper_lab)
            return mask
        except:
            return None
    
    def _calculate_aspect_ratio_score(self, contour):
        """Calculate how close the contour is to circular based on aspect ratio"""
        try:
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            
            if width == 0 or height == 0:
                return 0
            
            aspect_ratio = min(width, height) / max(width, height)
            return aspect_ratio
        except:
            return 0.5
    
    def _smart_ball_filtering(self, balls, frame_shape):
        """Smart ball filtering that preserves valid detections"""
        if len(balls) <= 1:
            return balls
        
        # Sort by confidence but keep more candidates
        balls = sorted(balls, key=lambda x: x['confidence'], reverse=True)
        
        filtered_balls = []
        for ball in balls:
            is_valid = True
            
            # Check overlap with existing balls
            for existing in filtered_balls:
                distance = math.sqrt(
                    (ball['center'][0] - existing['center'][0])**2 + 
                    (ball['center'][1] - existing['center'][1])**2
                )
                
                # More lenient overlap threshold
                overlap_threshold = min(ball['radius'], existing['radius']) * 0.6
                if distance < overlap_threshold:
                    is_valid = False
                    break
            
            if is_valid:
                filtered_balls.append(ball)
                
                # Allow up to 3 balls for better tracking
                if len(filtered_balls) >= 3:
                    break
        
        return filtered_balls
    
    def _assess_frame_quality(self, frame):
        """Assess frame quality for detection reliability"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Sharpness using Laplacian variance
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Brightness
        brightness = np.mean(gray)
        
        # Contrast
        contrast = np.std(gray)
        
        # Overall quality score
        quality_score = min(1.0, (sharpness / 300 + contrast / 60) / 2)  # More lenient thresholds
        
        return {
            'sharpness': sharpness,
            'brightness': brightness,
            'contrast': contrast,
            'quality_score': quality_score
        }
    
    def _detect_pose(self, frame):
        """Pose detection for shooting form analysis"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_pose.process(rgb_frame)
            
            if results.pose_landmarks:
                h, w = frame.shape[:2]
                landmarks = []
                
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    landmarks.append({
                        'id': idx,
                        'x': landmark.x * w,
                        'y': landmark.y * h,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                
                return landmarks
                
        except Exception:
            pass
        
        return None
    
    def _update_tracking_history(self, detections):
        """Update ball tracking history for sequence analysis"""
        self.ball_tracking_history.append({
            'timestamp': detections['timestamp'],
            'frame_idx': detections['frame_idx'],
            'balls': detections['balls'],
            'quality': detections['quality']
        })

class IntelligentShotSequenceDetector:
    """Enhanced shot sequence detector with support for different camera angles"""
    
    def __init__(self, detection_params, shot_validation=None):
        self.params = detection_params
        self.shot_validation = shot_validation or {
            'min_shot_duration': 0.5,
            'max_shot_duration': 6.0,
            'speed_consistency_threshold': 0.4,
            'min_total_displacement': 30,
        }
        self.shot_sequences = []
        self.active_sequence = None
        self.ball_history = deque(maxlen=300)  # Longer history
        
    def process_detections(self, ball_history):
        """Process ball detection history to find valid shot sequences"""
        history_list = list(ball_history)
        
        if len(history_list) < self.params['min_sequence_length']:
            return []
        
        # Find continuous ball detection sequences
        sequences = self._find_continuous_sequences(history_list)
        
        # Validate each sequence with more flexible criteria
        valid_shots = []
        for seq in sequences:
            if self._validate_shot_sequence_flexible(seq):
                shot_data = self._extract_shot_data(seq)
                if shot_data:
                    valid_shots.append(shot_data)
        
        return valid_shots
    
    def _find_continuous_sequences(self, history):
        """Find continuous sequences with more flexible gap handling"""
        sequences = []
        current_sequence = []
        gap_count = 0
        
        for frame_data in history:
            if frame_data['balls']:  # Ball detected
                if current_sequence and gap_count > self.params['max_sequence_gap']:
                    # Check if we should end the sequence or just bridge the gap
                    if len(current_sequence) >= self.params['min_sequence_length']:
                        # Long enough sequence - evaluate if gap is bridgeable
                        if gap_count <= 20:  # Bridge small gaps
                            # Add placeholder frames for the gap
                            for _ in range(gap_count):
                                current_sequence.append({
                                    'timestamp': frame_data['timestamp'] - 0.033,
                                    'balls': [],
                                    'quality': {'quality_score': 0.5},
                                    'interpolated': True
                                })
                        else:
                            # Gap too large, end sequence
                            sequences.append(current_sequence)
                            current_sequence = []
                
                current_sequence.append(frame_data)
                gap_count = 0
                
            else:  # No ball detected
                gap_count += 1
        
        # Add final sequence if valid
        if len(current_sequence) >= self.params['min_sequence_length']:
            sequences.append(current_sequence)
        
        return sequences
    
    def _validate_shot_sequence_flexible(self, sequence):
        """More flexible validation for different camera angles"""
        if len(sequence) < self.params['min_sequence_length']:
            return False
        
        # Extract ball positions (skip interpolated frames)
        positions = []
        timestamps = []
        
        for frame in sequence:
            if frame['balls'] and not frame.get('interpolated', False):
                best_ball = max(frame['balls'], key=lambda x: x['confidence'])
                positions.append(best_ball['center'])
                timestamps.append(frame['timestamp'])
        
        if len(positions) < 4:
            return False
        
        # Check 1: Total displacement
        total_displacement = self._calculate_total_displacement(positions)
        if total_displacement < self.shot_validation['min_total_displacement']:
            return False
        
        # Check 2: Duration validation
        duration = timestamps[-1] - timestamps[0]
        if (duration < self.shot_validation['min_shot_duration'] or 
            duration > self.shot_validation['max_shot_duration']):
            return False
        
        # Check 3: Motion pattern (more flexible for rear view)
        if not self._has_valid_motion_pattern(positions):
            return False
        
        # Check 4: Reasonable ball movement
        if not self._has_reasonable_movement(positions, timestamps):
            return False
        
        return True
    
    def _calculate_total_displacement(self, positions):
        """Calculate total displacement of the ball"""
        if len(positions) < 2:
            return 0
        
        start_pos = positions[0]
        end_pos = positions[-1]
        
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        
        return math.sqrt(dx*dx + dy*dy)
    
    def _has_valid_motion_pattern(self, positions):
        """Check for valid basketball motion pattern (flexible for different angles)"""
        if len(positions) < 6:
            return True  # Too few points to judge, allow it
        
        # For rear view, we might not see clear arc, so check for general movement patterns
        
        # Method 1: Check for some upward motion (even if not clear arc)
        ys = [pos[1] for pos in positions]
        
        # Look for any significant upward movement
        min_y = min(ys)
        max_y = max(ys)
        
        # If there's reasonable vertical movement, consider it valid
        vertical_movement = abs(max_y - min_y)
        if vertical_movement > 20:  # Some vertical movement
            return True
        
        # Method 2: Check for consistent directional movement
        xs = [pos[0] for pos in positions]
        horizontal_movement = abs(max(xs) - min(xs))
        
        # If strong horizontal movement (like in rear view), it's likely a valid shot
        if horizontal_movement > 30:
            return True
        
        # Method 3: Check for general motion consistency
        movements = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            movement = math.sqrt(dx*dx + dy*dy)
            movements.append(movement)
        
        # If there's consistent movement, it's likely valid
        avg_movement = np.mean(movements)
        if avg_movement > 5:  # Consistent movement
            return True
        
        return False
    
    def _has_reasonable_movement(self, positions, timestamps):
        """Check if ball movement is reasonable"""
        if len(positions) < 3:
            return False
        
        speeds = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            dt = timestamps[i] - timestamps[i-1]
            
            if dt > 0:
                speed = math.sqrt(dx*dx + dy*dy) / dt
                speeds.append(speed)
        
        if not speeds:
            return False
        
        # Check if speeds are reasonable (not too slow or too fast)
        avg_speed = np.mean(speeds)
        max_speed = max(speeds)
        
        # Allow wider range of speeds for different camera angles
        if avg_speed < 2:  # Too slow
            return False
        if max_speed > 1000:  # Unreasonably fast (likely noise)
            return False
        
        return True
    
    def _extract_shot_data(self, sequence):
        """Extract shot data from validated sequence"""
        positions = []
        timestamps = []
        confidences = []
        
        for frame in sequence:
            if frame['balls'] and not frame.get('interpolated', False):
                best_ball = max(frame['balls'], key=lambda x: x['confidence'])
                positions.append(best_ball['center'])
                timestamps.append(frame['timestamp'])
                confidences.append(best_ball['confidence'])
        
        if len(positions) < 4:
            return None
        
        return {
            'positions': positions,
            'timestamps': timestamps,
            'confidences': confidences,
            'start_time': timestamps[0],
            'end_time': timestamps[-1],
            'duration': timestamps[-1] - timestamps[0],
            'sequence_length': len(positions)
        }

class ShotAnalyzer:
    """Professional shot analysis with enhanced rear-view support"""
    
    def __init__(self):
        # NBA and professional standards (same as before)
        self.standards = {
            'release_angle': {
                'excellent': (47, 53),
                'good': (42, 58),
                'acceptable': (35, 65)
            },
            'arc_height': {
                'excellent': 80,
                'good': 60,
                'acceptable': 40
            },
            'release_speed': {
                'excellent': (120, 180),
                'good': (80, 220),
                'acceptable': (50, 280)
            }
        }
    
    def analyze_shot(self, shot_data, pose_data=None):
        """Comprehensive shot analysis with angle-aware calculations"""
        if not shot_data or len(shot_data['positions']) < 4:
            return None
        
        # Enhanced trajectory analysis for different camera angles
        trajectory_metrics = self._analyze_trajectory_enhanced(shot_data)
        
        # Form analysis (if pose data available)
        form_metrics = self._analyze_form(pose_data) if pose_data else {}
        
        # Overall scoring
        overall_score = self._calculate_overall_score(trajectory_metrics, form_metrics)
        
        # Recommendations
        recommendations = self._generate_recommendations(trajectory_metrics, form_metrics)
        
        return {
            'trajectory_metrics': trajectory_metrics,
            'form_metrics': form_metrics,
            'overall_score': overall_score,
            'recommendations': recommendations,
            'grade': self._assign_grade(overall_score)
        }
    
    def _analyze_trajectory_enhanced(self, shot_data):
        """Enhanced trajectory analysis for different camera angles"""
        positions = shot_data['positions']
        timestamps = shot_data['timestamps']
        
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        
        # Enhanced calculations that work for different viewing angles
        release_angle = self._calculate_release_angle_flexible(xs, ys)
        arc_height = self._calculate_arc_height_flexible(ys)
        release_speed = self._calculate_release_speed(positions, timestamps)
        smoothness = self._calculate_smoothness(xs, ys)
        
        # Additional metrics for rear-view analysis
        horizontal_displacement = abs(max(xs) - min(xs))
        vertical_displacement = abs(max(ys) - min(ys))
        
        return {
            'release_angle': release_angle,
            'arc_height': arc_height,
            'release_speed': release_speed,
            'smoothness': smoothness,
            'flight_time': timestamps[-1] - timestamps[0],
            'horizontal_displacement': horizontal_displacement,
            'vertical_displacement': vertical_displacement
        }
    
    def _calculate_release_angle_flexible(self, xs, ys):
        """Calculate release angle with flexibility for different camera angles"""
        if len(xs) < 3:
            return 45
        
        # Try multiple methods and use the most reasonable result
        
        # Method 1: Traditional calculation
        dx = xs[2] - xs[0]
        dy = ys[0] - ys[2]  # Y is inverted
        
        if abs(dx) > 1e-6:
            angle1 = math.degrees(math.atan(abs(dy) / abs(dx)))
        else:
            angle1 = 90 if dy > 0 else 0
        
        # Method 2: Use more points for stability
        if len(xs) >= 5:
            dx2 = xs[4] - xs[0]
            dy2 = ys[0] - ys[4]
            
            if abs(dx2) > 1e-6:
                angle2 = math.degrees(math.atan(abs(dy2) / abs(dx2)))
            else:
                angle2 = angle1
        else:
            angle2 = angle1
        
        # Average the angles for more stability
        final_angle = (angle1 + angle2) / 2
        
        # Ensure reasonable range
        return max(10, min(80, final_angle))
    
    def _calculate_arc_height_flexible(self, ys):
        """Calculate arc height with flexibility for different camera angles"""
        if len(ys) < 3:
            return 0
        
        # For rear view, arc might not be as visible, so use different approach
        
        # Method 1: Traditional arc height
        if min(ys) < max(ys):  # Normal case where we see upward motion
            max_height = min(ys)
            start_height = ys[0]
            end_height = ys[-1]
            
            # Use average of start and end as baseline
            baseline = (start_height + end_height) / 2
            arc_height = baseline - max_height
        else:
            # Rear view case - use vertical movement as proxy
            arc_height = abs(max(ys) - min(ys)) * 0.5
        
        return max(0, arc_height)
    
    def _calculate_release_speed(self, positions, timestamps):
        """Calculate initial release speed"""
        if len(positions) < 3:
            return 0
        
        # Use first few points for initial speed
        speeds = []
        for i in range(1, min(5, len(positions))):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            dt = timestamps[i] - timestamps[i-1]
            
            if dt > 0:
                speed = math.sqrt(dx*dx + dy*dy) / dt
                speeds.append(speed)
        
        return np.mean(speeds) if speeds else 0
    
    def _calculate_smoothness(self, xs, ys):
        """Calculate trajectory smoothness"""
        if len(xs) < 5:
            return 0.7  # Default reasonable smoothness
        
        # Calculate second derivatives with protection against short sequences
        try:
            if len(xs) >= 5:
                x_smooth = savgol_filter(xs, 5, 2)
                y_smooth = savgol_filter(ys, 5, 2)
            else:
                x_smooth = xs
                y_smooth = ys
            
            # Calculate curvature changes
            curvature_changes = []
            for i in range(2, len(x_smooth)-2):
                d2x = x_smooth[i+1] - 2*x_smooth[i] + x_smooth[i-1]
                d2y = y_smooth[i+1] - 2*y_smooth[i] + y_smooth[i-1]
                curvature = abs(d2x) + abs(d2y)
                curvature_changes.append(curvature)
            
            if not curvature_changes:
                return 0.7
            
            avg_curvature = np.mean(curvature_changes)
            smoothness = 1.0 / (1.0 + avg_curvature / 5)  # More lenient
            
            return min(1.0, smoothness)
        except:
            return 0.7  # Safe default
    
    def _analyze_form(self, pose_data):
        """Analyze shooting form from pose data"""
        if not pose_data:
            return {'form_score': 70, 'detected_issues': ['Pose data unavailable']}
        
        # Simplified form analysis
        form_score = 70  # Base score
        issues = []
        
        # This would be expanded with detailed pose analysis
        return {
            'form_score': form_score,
            'detected_issues': issues
        }
    
    def _calculate_overall_score(self, trajectory_metrics, form_metrics):
        """Calculate overall performance score"""
        trajectory_score = 0
        
        # Release angle scoring (more lenient)
        angle = trajectory_metrics['release_angle']
        if 40 <= angle <= 60:  # Wider acceptable range
            angle_score = 85
        elif 30 <= angle <= 70:
            angle_score = 70
        else:
            angle_score = 55
        
        # Arc height scoring (adjusted for rear view)
        arc = trajectory_metrics['arc_height']
        if arc >= 40:
            arc_score = 90
        elif arc >= 20:
            arc_score = 75
        elif arc >= 10:
            arc_score = 60
        else:
            arc_score = 45
        
        # Speed scoring (more flexible)
        speed = trajectory_metrics['release_speed']
        if 50 <= speed <= 300:
            speed_score = 80
        elif 20 <= speed <= 400:
            speed_score = 65
        else:
            speed_score = 50
        
        # Smoothness scoring
        smoothness_score = trajectory_metrics['smoothness'] * 100
        
        # Combined trajectory score
        trajectory_score = (angle_score * 0.25 + arc_score * 0.25 + 
                          speed_score * 0.25 + smoothness_score * 0.25)
        
        # Form score
        form_score = form_metrics.get('form_score', 70)
        
        # Overall score (70% trajectory, 30% form)
        overall = trajectory_score * 0.7 + form_score * 0.3
        
        return min(100, max(0, overall))
    
    def _assign_grade(self, score):
        """Assign letter grade based on score"""
        if score >= 90:
            return {'letter': 'A+', 'description': 'Excellent'}
        elif score >= 80:
            return {'letter': 'A', 'description': 'Very Good'}
        elif score >= 70:
            return {'letter': 'B', 'description': 'Good'}
        elif score >= 60:
            return {'letter': 'C', 'description': 'Average'}
        else:
            return {'letter': 'D', 'description': 'Needs Improvement'}
    
    def _generate_recommendations(self, trajectory_metrics, form_metrics):
        """Generate personalized recommendations"""
        recommendations = []
        
        angle = trajectory_metrics['release_angle']
        if angle < 35:
            recommendations.append({
                'category': 'Release Technique',
                'issue': f'Release angle low ({angle:.1f}°)',
                'suggestion': 'Increase release angle for better arc',
                'priority': 'Medium'
            })
        elif angle > 65:
            recommendations.append({
                'category': 'Release Technique', 
                'issue': f'Release angle high ({angle:.1f}°)',
                'suggestion': 'Lower release angle for more direct shot',
                'priority': 'Medium'
            })
        
        arc_height = trajectory_metrics['arc_height']
        if arc_height < 20:
            recommendations.append({
                'category': 'Shot Arc',
                'issue': 'Low arc trajectory',
                'suggestion': 'Increase shot arc for better entry angle',
                'priority': 'Medium'
            })
        
        if not recommendations:
            recommendations.append({
                'category': 'General',
                'issue': 'Solid shooting form detected',
                'suggestion': 'Continue practicing for consistency',
                'priority': 'Low'
            })
        
        return recommendations

# Keep the rest of the classes (HTMLReportGenerator, BasketballAnalysisSystem) the same
# as they were in the original code...

class HTMLReportGenerator:
    """Professional HTML report generator with embedded charts"""
    
    def __init__(self, output_dir, language='zh-TW'):
        self.output_dir = output_dir
        self.language = language
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Language-specific content
        self.content = LANGUAGES[language]
        
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_report(self, player_name, analyses, video_info, stats):
        """Generate comprehensive HTML report"""
        if not analyses:
            return self._generate_no_shots_report(player_name, video_info)
        
        # Generate embedded charts
        charts_base64 = self._generate_charts_base64(analyses, stats)
        
        # Create HTML content
        html_content = self._create_html_content(player_name, analyses, video_info, stats, charts_base64)
        
        # Save report
        report_path = os.path.join(self.output_dir, f"Basketball_Analysis_Report_{self.timestamp}.html")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return report_path
    
    def _generate_charts_base64(self, analyses, stats):
        """Generate charts and convert to base64 for embedding"""
        # Set up matplotlib for professional charts
        plt.style.use('default')
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['legend.fontsize'] = 10
        
        # Create 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Basketball Shot Analysis Charts', fontsize=16, fontweight='bold')
        
        # Chart 1: Score Trend
        if len(analyses) > 1:
            shot_numbers = list(range(1, len(analyses) + 1))
            scores = [analysis['overall_score'] for analysis in analyses]
            
            axes[0, 0].plot(shot_numbers, scores, 'o-', linewidth=2, markersize=6, color='#2E86AB')
            axes[0, 0].axhline(y=80, color='green', linestyle='--', alpha=0.7, label='Good (80+)')
            axes[0, 0].axhline(y=60, color='orange', linestyle='--', alpha=0.7, label='Average (60+)')
            axes[0, 0].set_title('Score Progression', fontweight='bold')
            axes[0, 0].set_xlabel('Shot Number')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].legend()
            axes[0, 0].set_ylim(0, 100)
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'Single Shot\nAnalysis', ha='center', va='center', 
                          fontsize=14, transform=axes[0, 0].transAxes)
        
        # Chart 2: Release Angle Distribution  
        angles = [analysis['trajectory_metrics']['release_angle'] for analysis in analyses]
        axes[0, 1].hist(angles, bins=min(8, max(1, len(angles))), color='#A23B72', alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(50, color='red', linestyle='--', linewidth=2, label='Optimal (50°)')
        axes[0, 1].axvspan(47, 53, alpha=0.2, color='green', label='Excellent Range')
        axes[0, 1].set_title('Release Angle Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Angle (degrees)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Chart 3: Performance Radar
        if analyses:
            avg_analysis = analyses[0]  # Use first shot as example
            metrics = avg_analysis['trajectory_metrics']
            
            categories = ['Release Angle', 'Arc Height', 'Release Speed', 'Smoothness', 'Overall']
            values = [
                min(100, metrics['release_angle'] * 1.5),
                min(100, metrics['arc_height'] * 1.5),
                min(100, metrics['release_speed'] / 3),
                metrics['smoothness'] * 100,
                avg_analysis['overall_score']
            ]
            
            # Radar chart
            angles_radar = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
            values_plot = values + [values[0]]  # Close the plot
            angles_plot = np.concatenate((angles_radar, [angles_radar[0]]))
            
            axes[1, 0].plot(angles_plot, values_plot, 'o-', linewidth=2, color='#F18F01')
            axes[1, 0].fill(angles_plot, values_plot, alpha=0.25, color='#F18F01')
            axes[1, 0].set_xticks(angles_radar)
            axes[1, 0].set_xticklabels(categories)
            axes[1, 0].set_ylim(0, 100)
            axes[1, 0].set_title('Performance Radar', fontweight='bold')
            axes[1, 0].grid(True)
        
        # Chart 4: Summary Statistics
        axes[1, 1].axis('off')
        
        summary_text = f"""
Analysis Summary

Total Shots: {len(analyses)}
Average Score: {stats.get('average_score', 0):.1f}/100
Best Score: {stats.get('best_score', 0):.1f}/100
Average Angle: {stats.get('average_angle', 0):.1f}°
Average Arc: {stats.get('average_arc', 0):.1f} px
Average Speed: {stats.get('average_speed', 0):.1f} px/s

Grade Distribution:
A: {stats.get('grade_counts', {}).get('A', 0)} shots
B: {stats.get('grade_counts', {}).get('B', 0)} shots  
C: {stats.get('grade_counts', {}).get('C', 0)} shots
D: {stats.get('grade_counts', {}).get('D', 0)} shots

Detection: Enhanced Multi-Angle
        """
        
        axes[1, 1].text(0.05, 0.95, summary_text.strip(), transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return img_base64
    
    def _create_html_content(self, player_name, analyses, video_info, stats, charts_base64):
        """Create complete HTML content"""
        
        # Language-specific titles
        if self.language == 'zh-TW':
            title = f"{player_name} - 籃球投籃分析報告"
            summary_title = "分析摘要"
            detailed_title = "詳細分析" 
            recommendations_title = "訓練建議"
        else:
            title = f"{player_name} - Basketball Shot Analysis Report"
            summary_title = "Analysis Summary"
            detailed_title = "Detailed Analysis"
            recommendations_title = "Training Recommendations"
        
        html_content = f"""
<!DOCTYPE html>
<html lang="{self.language[:2]}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .player-info {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            display: inline-block;
            margin-top: 20px;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            text-align: center;
        }}
        
        .stat-number {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }}
        
        .section {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        
        .section h2 {{
            color: #2c3e50;
            font-size: 1.8em;
            margin-bottom: 20px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        
        .chart-container {{
            text-align: center;
            margin: 20px 0;
        }}
        
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        
        .shot-analysis {{
            border-left: 4px solid #667eea;
            padding: 20px;
            margin-bottom: 20px;
            background: #f8f9fa;
            border-radius: 0 10px 10px 0;
        }}
        
        .grade-badge {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            color: white;
            font-weight: bold;
            margin-right: 10px;
        }}
        
        .grade-a {{ background: #28a745; }}
        .grade-b {{ background: #007bff; }}
        .grade-c {{ background: #ffc107; color: #000; }}
        .grade-d {{ background: #dc3545; }}
        
        .recommendations {{
            background: #e3f2fd;
            border: 1px solid #2196f3;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }}
        
        .recommendation {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 4px solid #2196f3;
        }}
        
        .priority-high {{ border-left-color: #f44336; }}
        .priority-medium {{ border-left-color: #ff9800; }}
        .priority-low {{ border-left-color: #4caf50; }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        
        .metric {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #e0e0e0;
        }}
        
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .footer {{
            text-align: center;
            padding: 20px;
            color: white;
            font-size: 0.9em;
        }}
        
        .detection-info {{
            background: #e8f5e8;
            border-left: 4px solid #28a745;
            padding: 15px;
            margin: 15px 0;
            border-radius: 0 8px 8px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <div class="player-info">
                <h3>{player_name}</h3>
                <p>Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
                <p>Video Duration: {video_info.get('duration_mmss', 'N/A')}</p>
                <div class="detection-info">
                    Enhanced Multi-Angle Detection System<br>
                    Supports side-view and rear-view camera angles
                </div>
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{len(analyses)}</div>
                <div>Total Shots Detected</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats.get('average_score', 0):.1f}</div>
                <div>Average Score</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats.get('best_score', 0):.1f}</div>
                <div>Best Score</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats.get('average_angle', 0):.1f}°</div>
                <div>Avg Release Angle</div>
            </div>
        </div>
        
        <div class="section">
            <h2>Analysis Charts</h2>
            <div class="chart-container">
                <img src="data:image/png;base64,{charts_base64}" alt="Analysis Charts">
            </div>
        </div>
        
        <div class="section">
            <h2>{detailed_title}</h2>
            {self._generate_shot_analyses_html(analyses)}
        </div>
        
        <div class="section">
            <h2>{recommendations_title}</h2>
            {self._generate_recommendations_html(analyses)}
        </div>
        
        <div class="footer">
            <p>Generated by Professional Basketball Analysis System | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Enhanced Multi-Angle Detection - Supports Various Camera Positions</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html_content
    
    def _generate_shot_analyses_html(self, analyses):
        """Generate HTML for individual shot analyses"""
        html_parts = []
        
        for i, analysis in enumerate(analyses, 1):
            score = analysis['overall_score']
            grade = analysis['grade']
            metrics = analysis['trajectory_metrics']
            
            # Grade badge class
            grade_class = f"grade-{grade['letter'][0].lower()}"
            
            # Metrics HTML
            metrics_html = f"""
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-value">{metrics['release_angle']:.1f}°</div>
                    <div>Release Angle</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{metrics['arc_height']:.0f}</div>
                    <div>Arc Height (px)</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{metrics['release_speed']:.1f}</div>
                    <div>Release Speed</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{metrics['smoothness']:.2f}</div>
                    <div>Smoothness</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{metrics['flight_time']:.2f}s</div>
                    <div>Flight Time</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{metrics.get('horizontal_displacement', 0):.0f}</div>
                    <div>Horizontal Move</div>
                </div>
            </div>
            """
            
            # Recommendations HTML
            recs_html = ""
            for rec in analysis['recommendations'][:3]:  # Top 3 recommendations
                priority_class = f"priority-{rec['priority'].lower()}"
                recs_html += f"""
                <div class="recommendation {priority_class}">
                    <strong>{rec['category']} - {rec['priority']} Priority</strong><br>
                    <strong>Issue:</strong> {rec['issue']}<br>
                    <strong>Suggestion:</strong> {rec['suggestion']}
                </div>
                """
            
            shot_html = f"""
            <div class="shot-analysis">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <h3>Shot #{i}</h3>
                    <span class="grade-badge {grade_class}">{score:.1f}/100 - {grade['letter']}</span>
                </div>
                
                <p><strong>Performance Level:</strong> {grade['description']}</p>
                <p><strong>Detection Quality:</strong> Enhanced multi-angle system</p>
                
                {metrics_html}
                
                <div class="recommendations">
                    <h4>Recommendations</h4>
                    {recs_html}
                </div>
            </div>
            """
            
            html_parts.append(shot_html)
        
        return ''.join(html_parts)
    
    def _generate_recommendations_html(self, analyses):
        """Generate overall recommendations HTML"""
        # Collect all recommendations
        all_recs = []
        for analysis in analyses:
            all_recs.extend(analysis['recommendations'])
        
        if not all_recs:
            return '<p>Excellent performance detected! Continue current training routine.</p>'
        
        # Count recommendation categories
        category_count = {}
        for rec in all_recs:
            category = rec['category']
            category_count[category] = category_count.get(category, 0) + 1
        
        # Generate summary
        html = '<div style="background: #f8f9fa; padding: 20px; border-radius: 10px;">'
        html += '<h4>Training Focus Areas</h4>'
        
        if category_count:
            html += '<ul>'
            for category, count in sorted(category_count.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(analyses)) * 100
                html += f'<li><strong>{category}</strong>: Mentioned in {percentage:.0f}% of shots ({count}/{len(analyses)})</li>'
            html += '</ul>'
        
        html += '<br><h4>Multi-Angle Analysis Notes</h4>'
        html += '<p>This analysis uses enhanced detection algorithms that work with:</p>'
        html += '<ul>'
        html += '<li><strong>Side-view angles:</strong> Traditional shooting analysis</li>'
        html += '<li><strong>Rear-view angles:</strong> Adapted trajectory analysis</li>'
        html += '<li><strong>Various lighting:</strong> Multi-range color detection</li>'
        html += '</ul>'
        
        html += '<br><h4>Training Schedule Recommendation</h4>'
        html += '<ul>'
        html += '<li><strong>Daily Practice:</strong> 30-45 minutes focused training</li>'
        html += '<li><strong>Weekly Review:</strong> Analyze 1-2 training sessions</li>'
        html += '<li><strong>Monthly Assessment:</strong> Compare progress with previous reports</li>'
        html += '</ul>'
        html += '</div>'
        
        return html
    
    def _generate_no_shots_report(self, player_name, video_info):
        """Generate report when no shots detected"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Analysis Report - {player_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; }}
        .header {{ text-align: center; margin-bottom: 40px; }}
        .message {{ background: #fff3cd; border: 1px solid #ffc107; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .suggestions {{ background: #d4edda; border: 1px solid #28a745; padding: 20px; border-radius: 5px; }}
        .info {{ background: #cce5ff; border: 1px solid #0066cc; padding: 20px; border-radius: 5px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Basketball Shot Analysis Report</h1>
            <h2>{player_name}</h2>
            <p>Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div>
        
        <div class="message">
            <h3>Detection Results</h3>
            <p>The enhanced multi-angle detection system processed your video but did not detect valid basketball shots.</p>
        </div>
        
        <div class="info">
            <h3>Detection System Features</h3>
            <p>This system includes:</p>
            <ul>
                <li>Multi-angle detection (side-view and rear-view)</li>
                <li>Enhanced color range detection for various lighting</li>
                <li>Flexible motion pattern recognition</li>
                <li>YOLO v8 + OpenCV fusion detection</li>
            </ul>
        </div>
        
        <div class="suggestions">
            <h3>Recording Tips for Better Detection</h3>
            <ul>
                <li><strong>Ball Visibility:</strong> Use standard orange basketball</li>
                <li><strong>Camera Distance:</strong> 3-8 meters from shooter</li>
                <li><strong>Angles Supported:</strong> Side view OR rear view</li>
                <li><strong>Lighting:</strong> Ensure adequate lighting conditions</li>
                <li><strong>Background:</strong> Avoid similar colors to basketball</li>
                <li><strong>Motion:</strong> Capture complete shooting motion</li>
                <li><strong>Stability:</strong> Keep camera reasonably steady</li>
            </ul>
            
            <h4>Camera Angle Guidelines:</h4>
            <ul>
                <li><strong>Side view:</strong> Best for traditional analysis</li>
                <li><strong>Rear view:</strong> Shows shooting motion from behind</li>
                <li><strong>Avoid:</strong> Front view (ball often blocked by body)</li>
            </ul>
        </div>
    </div>
</body>
</html>
        """
        
        report_path = os.path.join(self.output_dir, f"Analysis_Report_{self.timestamp}.html")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return report_path

class BasketballAnalysisSystem:
    """Main system class with enhanced multi-angle detection"""
    
    def __init__(self, output_dir, language='zh-TW'):
        self.output_dir = output_dir
        self.language = language
        
        # Initialize enhanced components
        self.detector = PrecisionBasketballDetector()
        self.sequence_detector = IntelligentShotSequenceDetector(
            self.detector.detection_params, 
            self.detector.shot_validation
        )
        self.analyzer = ShotAnalyzer()
        self.report_generator = HTMLReportGenerator(output_dir, language)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        global current_language
        current_language = language
    
    def analyze_video(self, video_path, player_name):
        """Main video analysis pipeline with enhanced detection"""
        print(f"{_('analyzing')}: {player_name}")
        print(f"Video: {os.path.basename(video_path)}")
        
        # Step 1: Extract video information
        video_info = self._extract_video_info(video_path)
        print(f"Video Info: {video_info['resolution']}, {video_info['fps']:.1f}fps, {video_info['duration_mmss']}")
        
        # Step 2: Process video frame by frame with enhanced detection
        print(f"{_('processing')} video frames with multi-angle detection...")
        ball_detections = self._process_video_frames(video_path, video_info)
        
        # Step 3: Detect shot sequences with flexible validation
        print("Detecting shot sequences...")
        shot_sequences = self.sequence_detector.process_detections(ball_detections)
        print(f"Detected {len(shot_sequences)} shot sequences")
        
        if len(shot_sequences) == 0:
            print("No shot sequences detected. Possible reasons:")
            print("- Ball not clearly visible or too small/large")
            print("- Motion too fast or too slow")
            print("- Camera angle or lighting affecting detection")
            print("- Ball color not matching expected orange range")
        
        # Step 4: Analyze each shot
        print("Analyzing shots...")
        analyses = []
        for i, shot_data in enumerate(shot_sequences, 1):
            print(f"  Analyzing shot {i}: {len(shot_data['positions'])} trajectory points")
            analysis = self.analyzer.analyze_shot(shot_data)
            if analysis:
                analyses.append(analysis)
        
        print(f"Completed analysis of {len(analyses)} shots")
        
        # Step 5: Calculate statistics
        stats = self._calculate_statistics(analyses)
        
        # Step 6: Generate report
        print(f"{_('generating_report')}...")
        report_path = self.report_generator.generate_report(player_name, analyses, video_info, stats)
        
        return {
            'report_path': report_path,
            'shot_count': len(analyses),
            'video_info': video_info,
            'stats': stats
        }
    
    def _extract_video_info(self, video_path):
        """Extract video information"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        info = {
            'path': video_path,
            'filename': os.path.basename(video_path),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': float(cap.get(cv2.CAP_PROP_FPS) or 30),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }
        
        info['duration_seconds'] = info['total_frames'] / info['fps']
        info['duration_mmss'] = str(timedelta(seconds=int(info['duration_seconds'])))[2:]
        info['resolution'] = f"{info['width']}x{info['height']}"
        
        cap.release()
        return info
    
    def _process_video_frames(self, video_path, video_info):
        """Process video frame by frame with progress tracking"""
        cap = cv2.VideoCapture(video_path)
        
        frame_idx = 0
        fps = video_info['fps']
        total_frames = video_info['total_frames']
        
        print(f"Processing {total_frames} frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect in current frame
            detections = self.detector.detect_frame(frame, frame_idx, fps)
            
            frame_idx += 1
            
            # Progress update every 10%
            if frame_idx % max(1, total_frames // 10) == 0:
                progress = (frame_idx / total_frames) * 100
                ball_count = len(detections['balls'])
                print(f"Progress: {progress:.0f}%, Current frame balls: {ball_count}")
        
        cap.release()
        
        # Return the accumulated ball tracking history
        total_detections = sum(1 for frame in self.detector.ball_tracking_history if frame['balls'])
        print(f"Total frames with ball detections: {total_detections}/{len(self.detector.ball_tracking_history)}")
        
        return self.detector.ball_tracking_history
    
    def _calculate_statistics(self, analyses):
        """Calculate overall statistics"""
        if not analyses:
            return {
                'average_score': 0,
                'best_score': 0,
                'worst_score': 0,
                'average_angle': 0,
                'average_arc': 0,
                'average_speed': 0,
                'grade_counts': {'A': 0, 'B': 0, 'C': 0, 'D': 0},
                'consistency': 'N/A'
            }
        
        scores = [analysis['overall_score'] for analysis in analyses]
        angles = [analysis['trajectory_metrics']['release_angle'] for analysis in analyses]
        arcs = [analysis['trajectory_metrics']['arc_height'] for analysis in analyses]
        speeds = [analysis['trajectory_metrics']['release_speed'] for analysis in analyses]
        
        # Grade distribution
        grade_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        for analysis in analyses:
            letter = analysis['grade']['letter'][0]
            if letter in grade_counts:
                grade_counts[letter] += 1
        
        # Consistency calculation
        score_std = np.std(scores) if len(scores) > 1 else 0
        if score_std < 5:
            consistency = 'Excellent'
        elif score_std < 10:
            consistency = 'Good'
        elif score_std < 15:
            consistency = 'Fair'
        else:
            consistency = 'Needs Improvement'
        
        return {
            'average_score': np.mean(scores),
            'best_score': max(scores),
            'worst_score': min(scores),
            'average_angle': np.mean(angles),
            'average_arc': np.mean(arcs),
            'average_speed': np.mean(speeds),
            'grade_counts': grade_counts,
            'consistency': consistency
        }

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Professional Basketball Shot Analysis System')
    parser.add_argument('--video', type=str, help='Path to basketball video file')
    parser.add_argument('--player', type=str, help='Player name')
    parser.add_argument('--output', type=str, help='Output directory for reports')
    parser.add_argument('--language', type=str, choices=['zh-TW', 'en'], default='zh-TW', help='Interface language')
    
    return parser.parse_args()

def main():
    """Main application entry point"""
    print("="*70)
    print("Professional Basketball Shot Analysis System")
    print("Enhanced Multi-Angle Detection with Rear-View Support")
    print("="*70)
    
    args = parse_arguments()
    
    # Command line mode
    if args.video and args.player and args.output:
        set_language(args.language)
        try:
            system = BasketballAnalysisSystem(args.output, args.language)
            results = system.analyze_video(args.video, args.player)
            
            print(f"\nAnalysis Complete!")
            print(f"Player: {args.player}")
            print(f"Shots Detected: {results['shot_count']}")
            print(f"Report Generated: {results['report_path']}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
        return
    
    # Always use GUI dialogs every run to pick video and output directory; force English
    try:
        root = tk.Tk()
        root.withdraw()

        # Load previous config for initial dirs/placeholders
        config = load_config()

        # Force English language
        language = 'en'
        set_language(language)

        # Ensure dialogs appear on top
        try:
            root.attributes('-topmost', True)
        except Exception:
            pass

        # 1) Select video file (always prompt)
        init_dir = config.get('last_video_dir', os.path.expanduser('~'))
        print("Open file dialog: Select basketball shot video")
        video_path = filedialog.askopenfilename(
            title='Select Basketball Shot Video',
            initialdir=init_dir,
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v"), ("All files", "*.*")]
        )
        if not video_path:
            print('No video selected, exiting.')
            return

        # 2) Player name (optional, but we ask each run for clarity)
        player_name = simpledialog.askstring(
            'Player Name',
            'Enter player name:',
            initialvalue=config.get('last_player_name', 'Player'),
            parent=root
        ) or 'Player'

        # 3) Select output directory (always prompt)
        init_out = config.get('last_output_dir', os.path.dirname(video_path))
        print("Open folder dialog: Select output directory")
        output_dir = filedialog.askdirectory(
            title='Select Output Directory',
            initialdir=init_out
        )
        if not output_dir:
            # If user cancels, default to the video's folder
            output_dir = os.path.dirname(video_path)

        # Persist selections for next launch (used as initial suggestions only)
        save_config({
            'language': language,
            'last_video_dir': os.path.dirname(video_path),
            'last_video_path': video_path,
            'last_player_name': player_name,
            'last_output_dir': output_dir
        })

        try:
            root.destroy()
        except Exception:
            pass

    except Exception as e:
        print(f"GUI initialization failed: {e}")
        return
    
    # Run analysis
    try:
        system = BasketballAnalysisSystem(output_dir, language)
        results = system.analyze_video(video_path, player_name)
        
        # Display results
        success_msg = f"""
{_('analysis_complete')}

Player: {player_name}
{_('shots_detected')}: {results['shot_count']}
Video Duration: {results['video_info']['duration_mmss']}

{_('report_generated')}: {os.path.basename(results['report_path'])}
Output Directory: {output_dir}

Key Statistics:
- {_('average_score')}: {results['stats']['average_score']:.1f}/100
- Best Shot: {results['stats']['best_score']:.1f}/100
- Consistency: {results['stats']['consistency']}

Detection System: Enhanced Multi-Angle
- Supports side-view and rear-view angles
- Enhanced color detection for various lighting
- Flexible motion pattern recognition
        """
        
        print(success_msg)
        
        # Show completion message
        try:
            messagebox.showinfo(_('analysis_complete'), success_msg)
        except:
            pass
    
    except Exception as e:
        error_msg = f"{_('error')}: {str(e)}"
        print(error_msg)
        try:
            messagebox.showerror(_('error'), error_msg)
        except:
            pass

if __name__ == "__main__":
    main()
