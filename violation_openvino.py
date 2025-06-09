# Violation detection logic for traffic monitoring

import cv2
import numpy as np
import time
import math
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Only traffic-related classes for detection
traffic_class_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
    'traffic light', 'stop sign', 'parking meter'
]

class OpenVINOViolationDetector:
    """
    OpenVINO-optimized traffic violation detection system.
    
    This implementation is designed for high-performance real-time processing
    with efficient vehicle tracking and violation detection algorithms.
    """
    def __init__(self, frame_rate: float = 30.0, config: Dict = None):
        """
        Initialize the violation detector.
        
        Args:
            frame_rate: Video frame rate for speed calculations
            config: Configuration dictionary for violation detection parameters
        """
        self.frame_rate = frame_rate
        
        # Violation tracking
        self.violation_history = []
        self.vehicle_tracks = {}  # Track ID -> track data
        self.next_track_id = 1
        
        # Traffic state tracking
        self.traffic_light_states = {}  # Position -> (color, timestamp)
        self.traffic_light_history = defaultdict(list)  # For state change detection
        self.stop_sign_positions = []
          # Configuration parameters
        default_config = {
            'red_light_grace_period': 1.0,  # seconds
            'stop_sign_stop_duration': 2.0,  # seconds required at stop
            'speed_limit_default': 50,  # km/h default speed limit
            'speed_tolerance': 5,  # km/h tolerance over limit
            'min_track_length': 5,  # minimum frames for reliable tracking
            'max_track_age': 60,  # maximum frames to keep track without detection
            'tracking_max_distance': 100,  # max pixels for track association
            'tracking_max_frames_lost': 30,  # max frames before removing track
            'traffic_light_detection_zone': 100,  # pixels around traffic light
        }
        
        # Merge with provided config
        self.config = default_config.copy()
        if config:
            self.config.update(config)
        
        # Performance tracking
        self.processing_times = deque(maxlen=100)
        self.detection_count = 0
        
        # Statistics
        self.stats = {
            'total_violations': 0,
            'red_light_violations': 0,
            'stop_sign_violations': 0,
            'speed_violations': 0,
            'lane_violations': 0,
            'tracked_vehicles': 0
        }
        logger.info("✅ OpenVINO Violation Detector initialized")
    
    def detect_violations(self, detections: List[Dict], frame: np.ndarray, 
                         frame_timestamp: float) -> List[Dict]:
        """
        Detect traffic violations in the current frame.
        
        Args:
            detections: List of detections from vehicle detector (can be NumPy array or list of dicts)
            frame: Current video frame
            frame_timestamp: Timestamp of the frame
            
        Returns:
            List of violation dictionaries
        """
        start_time = time.time()
        
        try:
            violations = []
            
            # Convert detections to proper format if needed
            if isinstance(detections, np.ndarray):
                print(f"🔄 Converting NumPy array detections ({detections.shape}) to dict format")
                detections = self._convert_detections_to_dicts(detections)
                print(f"✅ Converted to {len(detections)} detection dictionaries")
            
            # Debug: Validate detections format
            if detections and len(detections) > 0:
                first_det = detections[0]
                if not isinstance(first_det, dict):
                    print(f"❌ Warning: Expected dict, got {type(first_det)}")
                    return []
                else:
                    print(f"✅ Detections in correct dict format. Sample keys: {list(first_det.keys())}")
            
            # Update vehicle tracking
            self._update_vehicle_tracking(detections, frame_timestamp)
            
            # Update traffic state
            self._update_traffic_state(detections, frame_timestamp)
            
            # Check for violations
            violations.extend(self._detect_red_light_violations(detections, frame, frame_timestamp))
            violations.extend(self._detect_stop_sign_violations(detections, frame, frame_timestamp))
            violations.extend(self._detect_speed_violations(detections, frame, frame_timestamp))
            violations.extend(self._detect_lane_violations(detections, frame, frame_timestamp))
            
            # Update statistics
            self._update_statistics(violations)
            
            # Add processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Add metadata to violations
            for violation in violations:
                violation['detection_time'] = datetime.now()
                violation['frame_timestamp'] = frame_timestamp
                violation['processing_time'] = processing_time
            
            # Store in history
            self.violation_history.extend(violations)
            
            return violations
            
        except Exception as e:
            logger.error(f"❌ Violation detection failed: {e}")            
        return []
    
    def _convert_detections_to_dicts(self, detections_np: np.ndarray, class_names: List[str] = None) -> List[Dict]:
        """
        Convert NumPy array detections to list of dictionaries format
        
        Args:
            detections_np: NumPy array with shape [N, 6+] where each row is [x1, y1, x2, y2, confidence, class_id, ...]
            class_names: List of class names, defaults to COCO classes
            
        Returns:
            List of detection dictionaries
        """
        if class_names is None:
            class_names = traffic_class_names
        
        results = []
        for det in detections_np:
            if len(det) < 6:
                continue
                
            x1, y1, x2, y2, conf, cls_id = det[:6]
            cls_id = int(cls_id)
            
            # Get class name
            if cls_id < len(class_names):
                class_name = class_names[cls_id]
            else:
                class_name = f"class_{cls_id}"
            
            # Determine detection type
            vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
            traffic_sign_classes = ['traffic light', 'stop sign']
            
            if class_name in vehicle_classes:
                detection_type = 'vehicle'
            elif class_name in traffic_sign_classes:
                detection_type = 'traffic_sign'
            else:
                detection_type = 'other'
            
            results.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(conf),
                'class_id': cls_id,
                'class_name': class_name,
                'type': detection_type,
                'timestamp': time.time(),
                'frame_id': getattr(self, 'frame_count', 0),
                'license_plate': '',
                'traffic_light_color': 'unknown' if class_name == 'traffic light' else ''
            })
        
        return results
    
    def _update_vehicle_tracking(self, detections: List[Dict], timestamp: float):
        """
        Update vehicle tracking with current detections.
        
        Uses position-based association for efficient tracking without deep learning.
        """
        # Safety check: Ensure detections is in the correct format
        if not isinstance(detections, list):
            print(f"⚠️ Warning: Expected list of detections, got {type(detections)}")
            return
        
        if detections and not isinstance(detections[0], dict):
            print(f"⚠️ Warning: Expected dict detections, got {type(detections[0])}")
            return
        
        vehicle_detections = [d for d in detections if d['type'] == 'vehicle']
        
        # Update existing tracks
        updated_tracks = set()
        
        for detection in vehicle_detections:
            bbox = detection['bbox']
            center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            
            # Find closest existing track
            best_track_id = None
            best_distance = float('inf')
            
            for track_id, track_data in self.vehicle_tracks.items():
                if track_data['last_update'] < timestamp - 2.0:  # Skip old tracks
                    continue
                    
                last_center = track_data['positions'][-1] if track_data['positions'] else None
                if last_center:
                    distance = math.sqrt((center[0] - last_center[0])**2 + (center[1] - last_center[1])**2)
                    
                    # Check if distance is reasonable for vehicle movement
                    if distance < 100 and distance < best_distance:  # Max 100 pixels movement
                        best_distance = distance
                        best_track_id = track_id
            
            # Update existing track or create new one
            if best_track_id is not None:
                track_data = self.vehicle_tracks[best_track_id]
                track_data['positions'].append(center)
                track_data['timestamps'].append(timestamp)
                track_data['bboxes'].append(bbox)
                track_data['detections'].append(detection)
                track_data['last_update'] = timestamp
                updated_tracks.add(best_track_id)
                
                # Limit track length
                max_length = 60  # Keep last 60 positions
                if len(track_data['positions']) > max_length:
                    track_data['positions'] = track_data['positions'][-max_length:]
                    track_data['timestamps'] = track_data['timestamps'][-max_length:]
                    track_data['bboxes'] = track_data['bboxes'][-max_length:]
                    track_data['detections'] = track_data['detections'][-max_length:]
            else:
                # Create new track
                track_id = self.next_track_id
                self.next_track_id += 1
                
                self.vehicle_tracks[track_id] = {
                    'positions': [center],
                    'timestamps': [timestamp],
                    'bboxes': [bbox],
                    'detections': [detection],
                    'last_update': timestamp,
                    'violations': []
                }
                updated_tracks.add(track_id)
        
        # Remove old tracks
        tracks_to_remove = []
        for track_id, track_data in self.vehicle_tracks.items():
            if timestamp - track_data['last_update'] > 5.0:  # 5 seconds timeout
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.vehicle_tracks[track_id]
        
        # Update statistics
        self.stats['tracked_vehicles'] = len(self.vehicle_tracks)
    
    def _update_traffic_state(self, detections: List[Dict], timestamp: float):
        """Update traffic light states and stop sign positions."""
        for detection in detections:
            if detection.get('class_name') == 'traffic light':
                bbox = detection['bbox']
                position = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                color = detection.get('traffic_light_color', 'unknown')
                
                # Find existing traffic light or create new entry
                found_existing = False
                for pos, (_, last_timestamp) in list(self.traffic_light_states.items()):
                    distance = math.sqrt((position[0] - pos[0])**2 + (position[1] - pos[1])**2)
                    if distance < 50:  # Same traffic light if within 50 pixels
                        self.traffic_light_states[pos] = (color, timestamp)
                        found_existing = True
                        break
                
                if not found_existing:
                    self.traffic_light_states[position] = (color, timestamp)
            
            elif detection.get('class_name') == 'stop sign':
                bbox = detection['bbox']
                position = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                
                # Add to stop sign positions if not already present
                found_existing = False
                for pos in self.stop_sign_positions:
                    distance = math.sqrt((position[0] - pos[0])**2 + (position[1] - pos[1])**2)
                    if distance < 50:  # Same stop sign if within 50 pixels
                        found_existing = True
                        break
                
                if not found_existing:
                    self.stop_sign_positions.append(position)
        
        # Clean up old traffic light states
        current_time = timestamp
        positions_to_remove = []
        for position, (color, last_timestamp) in self.traffic_light_states.items():
            if current_time - last_timestamp > 10.0:  # Remove if not seen for 10 seconds
                positions_to_remove.append(position)
        
        for position in positions_to_remove:
            del self.traffic_light_states[position]
    
    def _detect_red_light_violations(self, detections: List[Dict], frame: np.ndarray, 
                                   timestamp: float) -> List[Dict]:
        """Detect red light violations."""
        violations = []
        
        # Find red traffic lights
        red_lights = []
        for position, (color, light_timestamp) in self.traffic_light_states.items():
            if color == 'red' and timestamp - light_timestamp < 2.0:  # Recent red light
                red_lights.append(position)
        
        if not red_lights:
            return violations
        
        # Check vehicles crossing red lights
        for track_id, track_data in self.vehicle_tracks.items():
            if len(track_data['positions']) < 3:  # Need at least 3 positions for movement
                continue
            
            current_pos = track_data['positions'][-1]
            previous_pos = track_data['positions'][-2]
            
            # Check if vehicle is moving towards or past red light
            for red_light_pos in red_lights:
                # Simple intersection zone check (in real implementation, use proper zones)
                distance_to_light = math.sqrt(
                    (current_pos[0] - red_light_pos[0])**2 + 
                    (current_pos[1] - red_light_pos[1])**2
                )
                
                prev_distance_to_light = math.sqrt(
                    (previous_pos[0] - red_light_pos[0])**2 + 
                    (previous_pos[1] - red_light_pos[1])**2
                )
                
                # Check if vehicle crossed the intersection zone during red light
                if (prev_distance_to_light > 150 and distance_to_light < 100 and 
                    distance_to_light < prev_distance_to_light):
                    
                    violation = {
                        'type': 'red_light_violation',
                        'vehicle_track_id': track_id,
                        'violation_position': current_pos,
                        'traffic_light_position': red_light_pos,
                        'severity': 'high',
                        'confidence': 0.9,
                        'description': f'Vehicle ran red light at position {current_pos}',
                        'vehicle_bbox': track_data['bboxes'][-1],
                        'timestamp': timestamp
                    }
                    violations.append(violation)
                    
                    # Add to track violations
                    track_data['violations'].append(violation)
        
        return violations
    
    def _detect_stop_sign_violations(self, detections: List[Dict], frame: np.ndarray, 
                                   timestamp: float) -> List[Dict]:
        """Detect stop sign violations."""
        violations = []
        
        if not self.stop_sign_positions:
            return violations
        
        # Check vehicles at stop signs
        for track_id, track_data in self.vehicle_tracks.items():
            if len(track_data['positions']) < 10:  # Need sufficient track history
                continue
            
            current_pos = track_data['positions'][-1]
            
            # Check if vehicle is near stop sign
            for stop_sign_pos in self.stop_sign_positions:
                distance_to_stop = math.sqrt(
                    (current_pos[0] - stop_sign_pos[0])**2 + 
                    (current_pos[1] - stop_sign_pos[1])**2
                )
                
                if distance_to_stop < 80:  # Within stop sign zone
                    # Check if vehicle came to a complete stop
                    stop_duration = self._calculate_stop_duration(track_data, stop_sign_pos)
                    
                    if stop_duration < self.config['stop_sign_stop_duration']:
                        # Check if this violation was already detected recently
                        recent_violation = False
                        for violation in track_data['violations'][-5:]:  # Check last 5 violations
                            if (violation.get('type') == 'stop_sign_violation' and 
                                timestamp - violation.get('timestamp', 0) < 5.0):
                                recent_violation = True
                                break
                        
                        if not recent_violation:
                            violation = {
                                'type': 'stop_sign_violation',
                                'vehicle_track_id': track_id,
                                'violation_position': current_pos,
                                'stop_sign_position': stop_sign_pos,
                                'stop_duration': stop_duration,
                                'required_duration': self.config['stop_sign_stop_duration'],
                                'severity': 'medium',
                                'confidence': 0.8,
                                'description': f'Vehicle failed to stop completely at stop sign (stopped for {stop_duration:.1f}s)',
                                'vehicle_bbox': track_data['bboxes'][-1],
                                'timestamp': timestamp
                            }
                            violations.append(violation)
                            track_data['violations'].append(violation)
        
        return violations
    
    def _calculate_stop_duration(self, track_data: Dict, stop_position: Tuple[int, int]) -> float:
        """Calculate how long a vehicle stopped near a stop sign."""
        positions = track_data['positions']
        timestamps = track_data['timestamps']
        
        if len(positions) < 2:
            return 0.0
        
        # Find positions near the stop sign
        stop_frames = []
        for i, pos in enumerate(positions[-20:]):  # Check last 20 positions
            distance = math.sqrt((pos[0] - stop_position[0])**2 + (pos[1] - stop_position[1])**2)
            if distance < 100:  # Near stop sign
                # Check if vehicle is stationary (movement < 10 pixels between frames)
                if i > 0:
                    prev_pos = positions[len(positions) - 20 + i - 1]
                    movement = math.sqrt((pos[0] - prev_pos[0])**2 + (pos[1] - prev_pos[1])**2)
                    if movement < 10:  # Stationary
                        stop_frames.append(len(positions) - 20 + i)
        
        if len(stop_frames) < 2:
            return 0.0
        
        # Calculate duration of longest continuous stop
        max_stop_duration = 0.0
        current_stop_start = None
        
        for i, frame_idx in enumerate(stop_frames):
            if current_stop_start is None:
                current_stop_start = frame_idx
            elif frame_idx - stop_frames[i-1] > 2:  # Gap in stop frames
                # Calculate previous stop duration
                stop_duration = (timestamps[stop_frames[i-1]] - timestamps[current_stop_start])
                max_stop_duration = max(max_stop_duration, stop_duration)
                current_stop_start = frame_idx
        
        # Check final stop duration
        if current_stop_start is not None:
            stop_duration = (timestamps[stop_frames[-1]] - timestamps[current_stop_start])
            max_stop_duration = max(max_stop_duration, stop_duration)
        
        return max_stop_duration
    
    def _detect_speed_violations(self, detections: List[Dict], frame: np.ndarray, 
                               timestamp: float) -> List[Dict]:
        """Detect speed violations based on vehicle tracking."""
        violations = []
        
        for track_id, track_data in self.vehicle_tracks.items():
            if len(track_data['positions']) < 10:  # Need sufficient data for speed calculation
                continue
            
            # Calculate speed over last few frames
            speed_kmh = self._calculate_vehicle_speed(track_data)
            
            if speed_kmh > self.config['speed_limit_default'] + self.config['speed_tolerance']:
                # Check if this violation was already detected recently
                recent_violation = False
                for violation in track_data['violations'][-3:]:  # Check last 3 violations
                    if (violation.get('type') == 'speed_violation' and 
                        timestamp - violation.get('timestamp', 0) < 3.0):
                        recent_violation = True
                        break
                
                if not recent_violation:
                    violation = {
                        'type': 'speed_violation',
                        'vehicle_track_id': track_id,
                        'violation_position': track_data['positions'][-1],
                        'measured_speed': speed_kmh,
                        'speed_limit': self.config['speed_limit_default'],
                        'excess_speed': speed_kmh - self.config['speed_limit_default'],
                        'severity': 'high' if speed_kmh > self.config['speed_limit_default'] + 20 else 'medium',
                        'confidence': 0.7,  # Lower confidence due to simplified speed calculation
                        'description': f'Vehicle exceeding speed limit: {speed_kmh:.1f} km/h in {self.config["speed_limit_default"]} km/h zone',
                        'vehicle_bbox': track_data['bboxes'][-1],
                        'timestamp': timestamp
                    }
                    violations.append(violation)
                    track_data['violations'].append(violation)
        
        return violations
    
    def _calculate_vehicle_speed(self, track_data: Dict) -> float:
        """
        Calculate vehicle speed in km/h based on position tracking.
        
        This is a simplified calculation that assumes:
        - Fixed camera position
        - Approximate pixel-to-meter conversion
        - Known frame rate
        """
        positions = track_data['positions']
        timestamps = track_data['timestamps']
        
        if len(positions) < 5:
            return 0.0
        
        # Use last 5 positions for speed calculation
        recent_positions = positions[-5:]
        recent_timestamps = timestamps[-5:]
        
        # Calculate total distance traveled
        total_distance_pixels = 0.0
        for i in range(1, len(recent_positions)):
            dx = recent_positions[i][0] - recent_positions[i-1][0]
            dy = recent_positions[i][1] - recent_positions[i-1][1]
            distance_pixels = math.sqrt(dx*dx + dy*dy)
            total_distance_pixels += distance_pixels
        
        # Calculate time elapsed
        time_elapsed = recent_timestamps[-1] - recent_timestamps[0]
        
        if time_elapsed <= 0:
            return 0.0
        
        # Convert to speed
        # Rough approximation: 1 pixel ≈ 0.1 meters (depends on camera setup)
        pixels_per_meter = 10.0  # Adjust based on camera calibration
        distance_meters = total_distance_pixels / pixels_per_meter
        speed_ms = distance_meters / time_elapsed
        speed_kmh = speed_ms * 3.6  # Convert m/s to km/h
        
        return speed_kmh
    
    def _detect_lane_violations(self, detections: List[Dict], frame: np.ndarray, 
                              timestamp: float) -> List[Dict]:
        """
        Detect lane violations (simplified implementation).
        
        In a full implementation, this would require lane detection and tracking.
        """
        violations = []
        
        # Simplified lane violation detection based on vehicle positions
        # This is a placeholder implementation
        frame_height, frame_width = frame.shape[:2]
        
        for track_id, track_data in self.vehicle_tracks.items():
            if len(track_data['positions']) < 5:
                continue
            
            current_pos = track_data['positions'][-1]
            
            # Simple boundary check (assuming road is in center of frame)
            # In reality, this would use proper lane detection
            road_left = frame_width * 0.1
            road_right = frame_width * 0.9
            
            if current_pos[0] < road_left or current_pos[0] > road_right:
                # Check if this violation was already detected recently
                recent_violation = False
                for violation in track_data['violations'][-3:]:
                    if (violation.get('type') == 'lane_violation' and 
                        timestamp - violation.get('timestamp', 0) < 2.0):
                        recent_violation = True
                        break
                
                if not recent_violation:
                    violation = {
                        'type': 'lane_violation',
                        'vehicle_track_id': track_id,
                        'violation_position': current_pos,
                        'severity': 'low',
                        'confidence': 0.5,  # Low confidence due to simplified detection
                        'description': 'Vehicle outside road boundaries',
                        'vehicle_bbox': track_data['bboxes'][-1],
                        'timestamp': timestamp
                    }
                    violations.append(violation)
                    track_data['violations'].append(violation)
        
        return violations
    
    def _update_statistics(self, violations: List[Dict]):
        """Update violation statistics."""
        for violation in violations:
            self.stats['total_violations'] += 1
            violation_type = violation.get('type', '')
            
            if 'red_light' in violation_type:
                self.stats['red_light_violations'] += 1
            elif 'stop_sign' in violation_type:
                self.stats['stop_sign_violations'] += 1
            elif 'speed' in violation_type:
                self.stats['speed_violations'] += 1
            elif 'lane' in violation_type:
                self.stats['lane_violations'] += 1
    
    def get_statistics(self) -> Dict:
        """Get current violation statistics."""
        stats = self.stats.copy()
        
        # Add performance metrics
        if self.processing_times:
            stats['avg_processing_time'] = np.mean(self.processing_times)
            stats['fps'] = 1.0 / np.mean(self.processing_times) if np.mean(self.processing_times) > 0 else 0
        
        # Add tracking metrics
        stats['active_tracks'] = len(self.vehicle_tracks)
        stats['traffic_lights_detected'] = len(self.traffic_light_states)
        stats['stop_signs_detected'] = len(self.stop_sign_positions)
        
        return stats
    
    def get_violation_history(self, limit: int = 100) -> List[Dict]:
        """Get recent violation history."""
        return self.violation_history[-limit:] if limit > 0 else self.violation_history
    
    def reset_statistics(self):
        """Reset violation statistics."""
        self.stats = {
            'total_violations': 0,
            'red_light_violations': 0,
            'stop_sign_violations': 0,
            'speed_violations': 0,
            'lane_violations': 0,
            'tracked_vehicles': 0
        }
        self.violation_history.clear()
        logger.info("✅ Violation statistics reset")
        
    def cleanup(self):
        """Clean up resources."""
        self.vehicle_tracks.clear()
        self.traffic_light_states.clear()
        self.stop_sign_positions.clear()
        logger.info("✅ OpenVINO Violation Detector cleanup completed")
    
    def get_violation_summary(self, time_window: float = 3600) -> Dict:
        """
        Get summary of violations in the specified time window
        
        Args:
            time_window: Time window in seconds (default: 1 hour)
            
        Returns:
            Summary dictionary
        """
        current_time = time.time()
        recent_violations = [
            v for v in self.violation_history 
            if current_time - v['timestamp'] <= time_window
        ]
        
        summary = {
            'total_violations': len(recent_violations),
            'by_type': defaultdict(int),
            'by_severity': defaultdict(int),
            'avg_confidence': 0,
            'time_window': time_window
        }
        
        if recent_violations:
            for violation in recent_violations:
                summary['by_type'][violation['type']] += 1
                summary['by_severity'][violation['severity']] += 1
            
            summary['avg_confidence'] = sum(v['confidence'] for v in recent_violations) / len(recent_violations)
        
        return dict(summary)
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if self.processing_times:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
        else:
            avg_time = 0
            fps = 0
        
        return {
            'avg_processing_time': avg_time * 1000,  # ms
            'fps': fps,
            'total_detections': self.detection_count,
            'total_violations': len(self.violation_history),
            'active_tracks': len(self.vehicle_tracks)
        }
    
    def reset_history(self):
        """Reset violation history and tracking data"""
        self.violation_history.clear()
        self.vehicle_tracks.clear()
        self.traffic_light_states.clear()
        self.traffic_light_history.clear()
        self.detection_count = 0
        logger.info("✅ Violation detector history reset")

# Convenience function for backward compatibility
def create_violation_detector(**kwargs) -> OpenVINOViolationDetector:
    """Create OpenVINO violation detector with default settings."""
    return OpenVINOViolationDetector(**kwargs)

# For compatibility with existing code
ViolationDetector = OpenVINOViolationDetector  # Alias for drop-in replacement
