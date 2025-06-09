# Helper functions for drawing, IoU, and other utilities

import cv2
import numpy as np
import pandas as pd
import time
import os
import base64
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import json
import io
from PIL import Image

def bbox_iou(box1, box2):
    """
    Calculate IoU (Intersection over Union) between two bounding boxes
    
    Args:
        box1: First bounding box in format [x1, y1, x2, y2]
        box2: Second bounding box in format [x1, y1, x2, y2]
        
    Returns:
        IoU score between 0 and 1
    """
    # Ensure boxes are in [x1, y1, x2, y2] format and have valid dimensions
    if len(box1) < 4 or len(box2) < 4:
        return 0.0
        
    # Convert to float and ensure x2 > x1 and y2 > y1
    x1_1, y1_1, x2_1, y2_1 = map(float, box1[:4])
    x1_2, y1_2, x2_2, y2_2 = map(float, box2[:4])
    
    if x2_1 <= x1_1 or y2_1 <= y1_1 or x2_2 <= x1_2 or y2_2 <= y1_2:
        return 0.0
    
    # Calculate area of each box
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    if area1 <= 0 or area2 <= 0:
        return 0.0
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0  # No intersection
    
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate IoU
    union_area = area1 + area2 - intersection_area
    
    if union_area <= 0:
        return 0.0
    
    iou = intersection_area / union_area
    return iou

# Color mapping for traffic-related classes only
COLORS = {
    'person': (255, 165, 0),         # Orange
    'bicycle': (255, 0, 255),        # Magenta
    'car': (0, 255, 0),              # Green
    'motorcycle': (255, 255, 0),     # Cyan
    'bus': (0, 0, 255),              # Red
    'truck': (0, 128, 255),          # Orange-Blue
    'traffic light': (0, 165, 255),  # Orange
    'stop sign': (0, 0, 139),        # Dark Red
    'parking meter': (128, 0, 128),  # Purple
    'default': (0, 255, 255)         # Yellow as default
}

VIOLATION_COLORS = {
    'red_light_violation': (0, 0, 255),    # Red
    'stop_sign_violation': (0, 100, 255),  # Orange-Red
    'speed_violation': (0, 255, 255),      # Yellow
    'lane_violation': (255, 0, 255),       # Magenta
}

def draw_detections(frame: np.ndarray, detections: List[Dict], 
                   draw_labels: bool = True, draw_confidence: bool = True) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on frame with enhanced robustness
    
    Args:
        frame: Input frame
        detections: List of detection dictionaries
        draw_labels: Whether to draw class labels
        draw_confidence: Whether to draw confidence scores
        
    Returns:
        Annotated frame
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        print("Warning: Invalid frame provided to draw_detections")
        return np.zeros((300, 300, 3), dtype=np.uint8)  # Return blank frame as fallback
        
    annotated_frame = frame.copy()
    
    # Handle case when detections is None or empty
    if detections is None or len(detections) == 0:
        return annotated_frame
    
    # Get frame dimensions for validation
    h, w = frame.shape[:2]
    
    for detection in detections:
        if not isinstance(detection, dict):
            continue
            
        try:
            # Skip detection if it doesn't have bbox or has invalid confidence
            if 'bbox' not in detection:
                continue
                
            # Skip if confidence is below threshold (don't rely on external filtering)
            confidence = detection.get('confidence', 0.0)
            if confidence < 0.01:  # Apply a minimal threshold to ensure we're not drawing noise
                continue
            
            bbox = detection['bbox']
            class_name = detection.get('class_name', 'unknown')
            class_id = detection.get('class_id', -1)
            
            # Get color for class
            color = get_enhanced_class_color(class_name, class_id)
            
            # Ensure bbox has enough coordinates and they are numeric values
            if len(bbox) < 4 or not all(isinstance(coord, (int, float)) for coord in bbox[:4]):
                continue
                
            # Convert coordinates to integers
            try:
                x1, y1, x2, y2 = map(int, bbox[:4])
            except (ValueError, TypeError):
                print(f"Warning: Invalid bbox format: {bbox}")
                continue
                
            # Validate coordinates are within frame bounds
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            # Ensure x2 > x1 and y2 > y1 (at least 1 pixel width/height)
            if x2 <= x1 or y2 <= y1:
                # Instead of skipping, fix the coordinates to ensure at least 1 pixel width/height
                x2 = max(x1 + 1, x2)
                y2 = max(y1 + 1, y2)
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            label_parts = []
            if draw_labels:
                # Display proper class name
                display_name = class_name.replace('_', ' ').title()
                label_parts.append(display_name)
            if draw_confidence:
                label_parts.append(f"{confidence:.2f}")
            
            # Add category indicator for clarity
            category = detection.get('category', 'other')
            if category != 'other':
                label_parts.append(f"[{category.upper()}]")
            
            # Draw license plate if available
            if 'license_plate' in detection and detection['license_plate']:
                plate_text = detection['license_plate'].get('text', 'Unknown')
                label_parts.append(f"Plate: {plate_text}")
            
            # Handle traffic light detection specially
            if detection.get('type') == 'traffic_sign' and detection.get('sign_type') == 'traffic_light':
                light_color = detection.get('color', 'unknown')
                
                # Add traffic light color to label
                if light_color != 'unknown':
                    # Set color indicator based on traffic light state
                    if light_color == 'red':
                        color_indicator = (0, 0, 255)  # Red
                        label_parts.append("🔴 RED")
                    elif light_color == 'yellow':
                        color_indicator = (0, 255, 255)  # Yellow
                        label_parts.append("🟡 YELLOW")
                    elif light_color == 'green':
                        color_indicator = (0, 255, 0)  # Green
                        label_parts.append("🟢 GREEN")
                
                # Draw traffic light visual indicator (circle with detected color)
                circle_y = y1 - 15
                circle_x = x1 + 10
                circle_radius = 10
                
                if light_color == 'red':
                    cv2.circle(annotated_frame, (circle_x, circle_y), circle_radius, (0, 0, 255), -1)
                elif light_color == 'yellow':
                    cv2.circle(annotated_frame, (circle_x, circle_y), circle_radius, (0, 255, 255), -1)
                elif light_color == 'green':
                    cv2.circle(annotated_frame, (circle_x, circle_y), circle_radius, (0, 255, 0), -1)
                else:
                    cv2.circle(annotated_frame, (circle_x, circle_y), circle_radius, (128, 128, 128), -1)
            
            # Draw label if we have any text
            if label_parts:
                label = " ".join(label_parts)
                
                try:
                    # Get text size for background
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    
                    # Ensure label position is within frame
                    text_x = max(0, min(x1, w - text_width))
                    text_y = max(text_height + 10, y1)
                    
                    # Draw label background (use colored background)
                    bg_color = tuple(int(c * 0.8) for c in color)  # Darker version of box color
                    cv2.rectangle(
                        annotated_frame,
                        (text_x, text_y - text_height - 10),
                        (text_x + text_width + 5, text_y),
                        bg_color,
                        -1
                    )
                    # Draw label text (white text on colored background)
                    cv2.putText(
                        annotated_frame,
                        label,
                        (text_x + 2, text_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),  # White text
                        2
                    )
                except Exception as e:
                    print(f"Error drawing label: {e}")
                    
        except Exception as e:
            print(f"Error drawing detection: {e}")
            continue
    
    return annotated_frame

def draw_violations(frame: np.ndarray, violations: List[Dict]) -> np.ndarray:
    """
    Draw violation indicators on frame with enhanced robustness
    
    Args:
        frame: Input frame
        violations: List of violation dictionaries
        
    Returns:
        Annotated frame with violations
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        print("Warning: Invalid frame provided to draw_violations")
        return np.zeros((300, 300, 3), dtype=np.uint8)  # Return blank frame as fallback
    
    # Handle case when violations is None or empty
    if violations is None or len(violations) == 0:
        return frame.copy()
        
    annotated_frame = frame.copy()
    h, w = frame.shape[:2]
    
    for violation in violations:
        if not isinstance(violation, dict):
            continue
            
        try:
            violation_type = violation.get('type', 'unknown')
            color = VIOLATION_COLORS.get(violation_type, (0, 0, 255))  # Default to red
            
            # Draw vehicle bbox if available
            bbox = None
            if 'vehicle_bbox' in violation:
                bbox = violation['vehicle_bbox']
            elif 'bbox' in violation:
                bbox = violation['bbox']
                
            if bbox and len(bbox) >= 4:
                # Ensure bbox coordinates are numeric
                if not all(isinstance(coord, (int, float)) for coord in bbox[:4]):
                    continue
                    
                try:
                    # Convert bbox coordinates to integers
                    x1, y1, x2, y2 = map(int, bbox[:4])
                except (ValueError, TypeError):
                    print(f"Warning: Invalid violation bbox format: {bbox}")
                    continue
                
                # Validate coordinates are within frame bounds
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))
                
                # Ensure x2 > x1 and y2 > y1 (at least 1 pixel width/height)
                if x2 <= x1 or y2 <= y1:
                    # Instead of skipping, fix the coordinates
                    x2 = max(x1 + 1, x2)
                    y2 = max(y1 + 1, y2)
                
                # Draw thicker red border for violations
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 4)
                
                # Add violation warning icon
                icon_x = max(0, min(x1 - 30, w-30))
                icon_y = max(30, min(y1 + 30, h-10))
                cv2.putText(annotated_frame, "⚠", (icon_x, icon_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            
            # Draw violation description
            description = violation.get('description', violation_type)
            severity = violation.get('severity', 'medium')
            
            # Position for violation text - ensure it's visible
            y_position = min(50 + (violations.index(violation) * 30), h - 40)
            
            # Draw violation text with background
            text = f"VIOLATION: {description} ({severity.upper()})"
            draw_text_with_background(annotated_frame, text, (10, y_position), color)
        except Exception as e:
            print(f"Error drawing violation: {e}")
    
    return annotated_frame

def get_enhanced_class_color(class_name: str, class_id: int) -> Tuple[int, int, int]:
    """
    Get color for class with enhanced mapping (traffic classes only)
    
    Args:
        class_name: Name of the detected class
        class_id: COCO class ID
        
    Returns:
        BGR color tuple
    """
    # Only traffic class IDs/colors
    enhanced_colors = {
        0: (255, 165, 0),      # person - Orange
        1: (255, 0, 255),      # bicycle - Magenta
        2: (0, 255, 0),        # car - Green
        3: (255, 255, 0),      # motorcycle - Cyan
        4: (0, 0, 255),        # bus - Red
        5: (0, 128, 255),      # truck - Orange-Blue
        6: (0, 165, 255),      # traffic light - Orange
        7: (0, 0, 139),        # stop sign - Dark Red
        8: (128, 0, 128),      # parking meter - Purple
    }
    
    # Get color from class name if available
    if class_name.lower() in COLORS:
        return COLORS[class_name.lower()]
    
    # Get color from class ID if available
    if class_id in enhanced_colors:
        return enhanced_colors[class_id]
    
    # Default color
    return COLORS['default']

def draw_text_with_background(frame: np.ndarray, text: str, position: Tuple[int, int], 
                             color: Tuple[int, int, int], alpha: float = 0.7) -> np.ndarray:
    """
    Draw text with semi-transparent background
    
    Args:
        frame: Input frame
        text: Text to display
        position: Position (x, y) to display text
        color: Color for text and border
        alpha: Background transparency (0-1)
        
    Returns:
        Frame with text
    """
    x, y = position
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    )
    
    # Create background
    bg_color = (30, 30, 30)  # Dark background
    cv2.rectangle(
        frame,
        (x, y - text_height - 10),
        (x + text_width + 10, y + 10),
        bg_color,
        -1
    )
    
    # Add colored border
    cv2.rectangle(
        frame,
        (x, y - text_height - 10),
        (x + text_width + 10, y + 10),
        color,
        2
    )
    
    # Add text
    cv2.putText(
        frame,
        text,
        (x + 5, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )
    
    return frame

def create_detection_summary(detections: List[Dict]) -> Dict[str, int]:
    """
    Create summary of detections
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        Dictionary with detection counts by type
    """
    summary = {
        'total': len(detections),
        'vehicles': 0,
        'pedestrians': 0,
        'traffic_signs': 0,
        'bicycles': 0,
        'motorcycles': 0,
        'license_plates': 0
    }
    
    vehicle_types = {}
    
    for detection in detections:
        detection_type = detection.get('type', '')
        class_name = detection.get('class_name', '').lower()
        
        if detection_type == 'vehicle':
            summary['vehicles'] += 1
            vehicle_type = detection.get('vehicle_type', 'unknown')
            vehicle_types[vehicle_type] = vehicle_types.get(vehicle_type, 0) + 1
            
            # Count license plates
            if detection.get('license_plate'):
                summary['license_plates'] += 1
                
        elif 'person' in class_name:
            summary['pedestrians'] += 1
        elif detection_type == 'traffic_sign':
            summary['traffic_signs'] += 1
        elif 'bicycle' in class_name:
            summary['bicycles'] += 1
        elif 'motorcycle' in class_name:
            summary['motorcycles'] += 1
    
    # Add vehicle type breakdowns
    for vehicle_type, count in vehicle_types.items():
        summary[f"vehicle_{vehicle_type}"] = count
    
    return summary

def create_performance_metrics(detector, violation_detector) -> Dict[str, Any]:
    """
    Create performance metrics
    
    Args:
        detector: Vehicle detector instance
        violation_detector: Violation detector instance
        
    Returns:
        Dictionary with performance metrics
    """
    metrics = {}
    
    # Detection metrics
    if detector:
        try:
            # Try to get detector metrics
            if hasattr(detector, 'get_performance_stats'):
                stats = detector.get_performance_stats()
                metrics.update({
                    'fps': stats.get('fps', 0),
                    'inference_time': stats.get('avg_inference_time', 0) * 1000,
                    'frames_processed': stats.get('frames_processed', 0)
                })
            
            # Add detection count
            metrics['detection_count'] = getattr(detector, 'detection_count', 0)
        except Exception:
            pass
    
    # Violation metrics
    if violation_detector:
        try:
            # Count violations
            violation_count = len(violation_detector.violation_history)
            metrics['violation_count'] = violation_count
        except Exception:
            pass
    
    # Add session metrics
    try:
        import streamlit as st
        if 'start_time' in st.session_state:
            uptime = time.time() - st.session_state.start_time
            metrics['uptime'] = f"{uptime/3600:.1f}h"
        
        if 'processed_frames' in st.session_state:
            metrics['processed_frames'] = st.session_state.processed_frames
    except ImportError:
        # Streamlit not available
        pass
    
    return metrics

def export_detections_to_csv(detection_history: List[List[Dict]]) -> str:
    """
    Export detection history to CSV
    
    Args:
        detection_history: List of frame detection lists
        
    Returns:
        CSV string
    """
    records = []
    
    for frame_idx, frame_detections in enumerate(detection_history):
        for detection in frame_detections:
            record = {
                'frame_id': frame_idx,
                'timestamp': detection.get('timestamp', ''),
                'class_name': detection.get('class_name', ''),
                'confidence': detection.get('confidence', 0),
                'bbox_x1': detection['bbox'][0] if 'bbox' in detection else 0,
                'bbox_y1': detection['bbox'][1] if 'bbox' in detection else 0,
                'bbox_x2': detection['bbox'][2] if 'bbox' in detection else 0,
                'bbox_y2': detection['bbox'][3] if 'bbox' in detection else 0,
                'type': detection.get('type', ''),
                'vehicle_type': detection.get('vehicle_type', ''),
                'license_plate': detection.get('license_plate', {}).get('text', '') if detection.get('license_plate') else ''
            }
            records.append(record)
    
    # Convert to DataFrame and then CSV
    if records:
        df = pd.DataFrame(records)
        return df.to_csv(index=False)
    else:
        return "No data available"

def save_annotated_frame(frame: np.ndarray, suffix: str = None) -> str:
    """
    Save annotated frame to temp file
    
    Args:
        frame: Frame to save
        suffix: Optional filename suffix
        
    Returns:
        Path to saved file
    """
    import tempfile
    
    timestamp = int(time.time())
    suffix = f"_{suffix}" if suffix else ""
    filename = f"traffic_frame_{timestamp}{suffix}.jpg"
    filepath = os.path.join(tempfile.gettempdir(), filename)
    
    cv2.imwrite(filepath, frame)
    return filepath

def resize_frame_for_display(frame: np.ndarray, max_width: int = 800) -> np.ndarray:
    """
    Resize frame for display while maintaining aspect ratio
    
    Args:
        frame: Input frame
        max_width: Maximum display width
        
    Returns:
        Resized frame
    """
    height, width = frame.shape[:2]
    
    # Only resize if width exceeds max_width
    if width > max_width:
        ratio = max_width / width
        new_height = int(height * ratio)
        return cv2.resize(frame, (max_width, new_height))
    
    return frame

def load_configuration(config_file: str = "config.json") -> Dict:
    """
    Load application configuration
    
    Args:
        config_file: Configuration file path
        
    Returns:
        Configuration dictionary
    """
    default_config = {
        "detection": {
            "confidence_threshold": 0.4,
            "enable_ocr": True,
            "enable_tracking": True
        },
        "violations": {
            "red_light_grace_period": 2.0,
            "stop_sign_duration": 3.0,
            "speed_tolerance": 10
        },
        "display": {
            "show_confidence": True,
            "show_labels": True,
            "show_license_plates": True,
            "max_display_width": 800
        }
    }
    
    # Try to load existing configuration
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except Exception:
        # Return default if loading fails
        return default_config

def save_configuration(config: Dict, config_file: str = "config.json"):
    """
    Save application configuration
    
    Args:
        config: Configuration dictionary
        config_file: Configuration file path
    """
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)

class StreamlitUtils:
    """Utility methods for Streamlit UI"""
    
    @staticmethod
    def display_metrics(metrics: Dict, col1, col2, col3, col4):
        """
        Display metrics in columns
        
        Args:
            metrics: Dictionary of metrics
            col1, col2, col3, col4: Streamlit columns
        """
        try:
            import streamlit as st
            
            # First column - Detection counts
            with col1:
                if 'detection_count' in metrics:
                    st.metric("Detections", metrics['detection_count'])
                elif 'total_detections' in metrics:
                    st.metric("Detections", metrics['total_detections'])
            
            # Second column - Violation counts
            with col2:
                if 'violation_count' in metrics:
                    st.metric("Violations", metrics['violation_count'])
            
            # Third column - Performance
            with col3:
                if 'fps' in metrics:
                    st.metric("FPS", f"{metrics['fps']:.2f}")
                elif 'processing_fps' in metrics:
                    st.metric("Processing FPS", f"{metrics['processing_fps']:.2f}")
            
            # Fourth column - Status
            with col4:
                if 'uptime' in metrics:
                    st.metric("Uptime", metrics['uptime'])
                elif 'frames_processed' in metrics:
                    st.metric("Frames", metrics['frames_processed'])
        except ImportError:
            # Streamlit not available
            pass
    
    @staticmethod
    def display_detection_summary(summary: Dict):
        """
        Display detection summary
        
        Args:
            summary: Detection summary dictionary
        """
        try:
            import streamlit as st
            
            if not summary or summary.get('total', 0) == 0:
                st.info("No detections to display.")
                return
            
            # Create summary table
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Objects", summary['total'])
                st.metric("Vehicles", summary['vehicles'])
                st.metric("Pedestrians", summary['pedestrians'])
                
            with col2:
                st.metric("Traffic Signs", summary['traffic_signs'])
                st.metric("License Plates", summary['license_plates'])
                
                # Performance indicator
                if 'vehicles' in summary and summary['vehicles'] > 0:
                    license_rate = summary['license_plates'] / summary['vehicles'] * 100
                    st.metric("Plate Detection", f"{license_rate:.1f}%")
        except ImportError:
            # Streamlit not available
            pass
    
    @staticmethod
    def display_violation_alerts(violations: List[Dict]):
        """
        Display violation alerts
        
        Args:
            violations: List of violation dictionaries
        """
        try:
            import streamlit as st
            
            if not violations:
                st.info("No violations detected.")
                return
            
            for violation in violations:
                violation_type = violation['type']
                severity = violation.get('severity', 'medium')
                description = violation.get('description', violation_type)
                
                # Format violation alert based on severity
                if severity == 'high':
                    alert_icon = "🔴"
                    alert_color = "red"
                elif severity == 'medium':
                    alert_icon = "🟡"
                    alert_color = "orange"
                else:
                    alert_icon = "🟢"
                    alert_color = "green"
                
                # Display alert
                st.markdown(
                    f"<div class='violation-alert' style='border-color: {alert_color};'>"
                    f"<h3>{alert_icon} {violation_type.replace('_', ' ').title()}</h3>"
                    f"<p>{description}</p>"
                    f"<p><strong>Severity:</strong> {severity.upper()}</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        except ImportError:
            # Streamlit not available
            pass
    
    @staticmethod
    def create_download_button(data: Any, file_name: str, button_text: str, mime_type: str = "text/csv"):
        """
        Create file download button
        
        Args:
            data: File data
            file_name: Download file name
            button_text: Button label text
            mime_type: MIME type of file
        """
        try:
            import streamlit as st
            
            if isinstance(data, str):
                data = data.encode()
            
            b64 = base64.b64encode(data).decode()
            href = f'data:{mime_type};base64,{b64}'
            
            st.download_button(
                label=button_text,
                data=data,
                file_name=file_name,
                mime=mime_type
            )
        except ImportError:
            # Streamlit not available
            pass

def bbox_iou(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        boxA: First bounding box (x1, y1, x2, y2)
        boxB: Second bounding box (x1, y1, x2, y2)
        
    Returns:
        IoU value between 0 and 1
    """
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # Compute the area of intersection
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    # Compute the IoU
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    
    return iou

__all__ = [
    "draw_detections", "draw_violations", "create_performance_metrics",
    "load_configuration", "save_configuration", "StreamlitUtils",
    "bbox_iou"
]
