# Streamlit app for real-time traffic monitoring using OpenVINO
# Provides detection, violation monitoring, and analytics dashboard

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
import tempfile
import os
import sys
from pathlib import Path
import threading
import queue
import json
import os
import base64
from typing import Dict, List, Optional, Any
import warnings

warnings.filterwarnings('ignore')

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import custom modules
try:
    # Use OpenVINO-optimized detection and violation modules
    from detection_openvino import OpenVINOVehicleDetector
    from violation_openvino import OpenVINOViolationDetector
    from utils import (
        draw_detections, draw_violations, create_detection_summary,
        create_performance_metrics, export_detections_to_csv,
        save_annotated_frame, resize_frame_for_display,
        StreamlitUtils, load_configuration, save_configuration,
        bbox_iou
    )
    from annotation_utils import enhanced_annotate_frame
    OPTIMIZED_DETECTION = True
    print("✅ OpenVINO detection and violation modules loaded successfully!")
except ImportError as e:
    st.error(f"Error importing OpenVINO modules: {e}")
    st.stop()

# Try to import DeepSort
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False

# Add after the imports section and before the TrafficMonitoringApp class

import asyncio
import platform

# Fix asyncio event loop issue on Windows with Streamlit
def setup_asyncio():
    """Setup asyncio event loop for Streamlit compatibility"""
    try:
        if platform.system() == 'Windows':
            # Use ProactorEventLoop on Windows for better compatibility
            loop = asyncio.ProactorEventLoop()
            asyncio.set_event_loop(loop)
        else:
            # Use default event loop on other platforms
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
    except Exception as e:
        print(f"Warning: Could not setup asyncio event loop: {e}")

def find_best_model_path(base_model_name: str = "yolo11x", search_dirs: List[str] = None) -> Optional[str]:
    """
    Intelligently find the best available model file (.xml or .pt) in the workspace.
    
    Args:
        base_model_name: Base model name without extension
        search_dirs: Directories to search in. If None, uses default search locations.
    
    Returns:
        Path to the best available model file, or None if not found
    """
    if search_dirs is None:
        search_dirs = [
            ".",  # Current directory
            "rcb",  # RCB directory
            "models",  # Common models directory
            "weights",  # Common weights directory
        ]
    
    # Priority order: OpenVINO IR (.xml) > PyTorch (.pt)
    model_extensions = [
        (f"{base_model_name}_openvino_model/{base_model_name}.xml", "OpenVINO IR"),
        (f"{base_model_name}.xml", "OpenVINO IR"), 
        (f"{base_model_name}_openvino_model.xml", "OpenVINO IR"),
        (f"{base_model_name}.pt", "PyTorch"),
        (f"{base_model_name}.pth", "PyTorch"),
    ]
    
    found_models = []
    
    for search_dir in search_dirs:
        search_path = Path(search_dir)
        if not search_path.exists():
            continue
            
        for model_file, model_type in model_extensions:
            model_path = search_path / model_file
            if model_path.exists():
                abs_path = os.path.abspath(model_path)
                found_models.append((abs_path, model_type))
                print(f"✅ Found {model_type} model: {abs_path}")
    
    if found_models:
        # Return the first found model (priority order)
        best_model, model_type = found_models[0]
        print(f"🎯 Selected {model_type} model: {best_model}")
        return best_model
    
    print(f"❌ No model files found for '{base_model_name}' in directories: {search_dirs}")
    return None

def load_model_dynamically(model_name: str = "yolo11x", **detector_kwargs) -> Optional[OpenVINOVehicleDetector]:
    """
    Dynamically load model with intelligent file detection and format handling.
    
    Args:
        model_name: Base model name to search for
        **detector_kwargs: Additional arguments for OpenVINOVehicleDetector
    
    Returns:
        Initialized OpenVINOVehicleDetector or None if failed
    """
    try:
        # Find the best available model
        model_path = find_best_model_path(model_name)
        if not model_path:
            st.error(f"❌ Could not find any model files for '{model_name}'")
            return None
        
        # Determine model type and setup appropriate parameters
        # (Remove st.info and st.success here to avoid duplicate messages)
        
        # Initialize detector with the found model
        detector = OpenVINOVehicleDetector(
            model_path=model_path,
            **detector_kwargs
        )
        
        return detector
        
    except Exception as e:
        st.error(f"❌ Error loading model dynamically: {e}")
        print(f"Full error details: {e}")
        import traceback
        traceback.print_exc()
        return None

# Setup asyncio when module is imported
setup_asyncio()

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(145deg, #f0f2f6, #ffffff);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2E86AB;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .violation-alert {
        background: linear-gradient(145deg, #fff5f5, #fed7d7);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #e53e3e;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .detection-summary {
        background: linear-gradient(145deg, #f0fff4, #c6f6d5);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #38a169;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .openvino-status {
        background: linear-gradient(145deg, #ebf8ff, #bee3f8);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3182ce;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background: linear-gradient(145deg, #2E86AB, #1e6091);
        color: white;
        border: none;
        border-radius: 5px;
        font-weight: bold;
    }
    .stSelectbox > div > div {
        background: #f8f9fa;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

class TrafficMonitoringApp:
    """Main Traffic Monitoring Application with OpenVINO acceleration"""
    
    def __init__(self):
        """Initialize the application"""
        self.detector = None
        self.violation_detector = None
        self.config = self._load_default_config()
        self.detection_history = []
        self.violation_history = []
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=10)
        
        # Initialize session state
        self._initialize_session_state()
        
        # Load models
        self._load_models()
        
        # Initialize DeepSORT tracker if available
        if DEEPSORT_AVAILABLE:
            self.tracker = DeepSort(max_age=30, n_init=3, max_cosine_distance=0.2)
        else:
            self.tracker = None
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        session_vars = {
            'detection_count': 0,
            'violation_count': 0,
            'start_time': time.time(),
            'processed_frames': 0,
            'performance_stats': {},
            'detector': None,
            'violation_detector': None,
            'current_backend': 'CPU',
            'optimization_active': False
        }
        
        for var, default_value in session_vars.items():
            if var not in st.session_state:
                st.session_state[var] = default_value
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""         
        return {
            'detection': {
                'confidence_threshold': 0.4,  # Higher threshold to prevent over-detection
                'enable_ocr': True,
                'enable_tracking': True,
                'device': 'AUTO',  # OpenVINO device selection
                'enable_int8': False,  # INT8 quantization
                'async_inference': True
            },
            'violations': {
                'red_light_grace_period': 2.0,
                'stop_sign_duration': 2.0,
                'speed_tolerance': 10,
                'enable_tracking': True
            },
            'display': {
                'show_confidence': True,
                'show_labels': True,
                'show_license_plates': True,
                'max_display_width': 800,
                'show_performance': True
            },
            'performance': {
                'max_history_size': 1000,
                'frame_skip': 1,
                'enable_gpu': True
            }
        }
    @st.cache_resource
    def _load_models(_self):
        """Load OpenVINO-optimized models with dynamic model detection"""
        try:
            with st.spinner("🚀 Loading OpenVINO-optimized models..."):
                # Use consistent confidence threshold for both detection and display
                detection_threshold = _self.config['detection']['confidence_threshold']
                
                # Use dynamic model loading
                detector = load_model_dynamically(
                    model_name="yolo11x",
                    device=_self.config['detection']['device'],
                    use_quantized=_self.config['detection']['enable_int8'],
                    enable_ocr=_self.config['detection']['enable_ocr'],
                    confidence_threshold=detection_threshold  # Use the same threshold value
                )
                
                if detector is None:
                    st.error("❌ Failed to load vehicle detection model")
                    return None, None
                
                # Initialize violation detector
                violation_config = {
                    'min_track_length': 10 if _self.config['violations']['enable_tracking'] else 5
                }
                violation_detector = OpenVINOViolationDetector(
                    config=violation_config
                )
                
                # Store in session state
                st.session_state.detector = detector
                st.session_state.violation_detector = violation_detector
                st.session_state.optimization_active = True
                st.session_state.current_backend = detector.device
                
                # st.success(f"✅ OpenVINO models loaded successfully! Device: {detector.device}")
                return detector, violation_detector
                
        except Exception as e:
            st.error(f"❌ Error loading OpenVINO models: {e}")
            print(f"Full error details: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def run(self):
        """Main application entry point"""
        # Auto-reload model if missing from session state (for Streamlit refresh)
        if ("detector" not in st.session_state or st.session_state.detector is None):
            detector, violation_detector = self._load_models()
            if detector is not None:
                st.session_state.detector = detector
                st.session_state.violation_detector = violation_detector
            else:
                st.stop()
        self.detector = st.session_state.detector
        self.violation_detector = st.session_state.violation_detector
        # Header with OpenVINO status
        self._render_header()
        
        # Sidebar configuration
        self._render_sidebar()
        
        # Main content area
        self._render_main_content()
    
    def _render_header(self):
        """Render application header with OpenVINO status"""
        header_col1, header_col2 = st.columns([3, 1])
        with header_col1:
            st.markdown(
                '<h1 class="main-header">🚦 Advanced Traffic Monitoring with OpenVINO</h1>', 
                unsafe_allow_html=True
            )
        with header_col2:
            if "detector" in st.session_state and st.session_state.detector is not None:
                st.markdown(
                    f'<div class="openvino-status">🚀 OpenVINO Active<br/>Device: {getattr(st.session_state.detector, "device", "AUTO")}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.warning("⚠️ OpenVINO not loaded")
    
    def _render_sidebar(self):
        """Render sidebar configuration"""
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # OpenVINO Settings
            with st.expander("🚀 OpenVINO Settings", expanded=True):
                device_options = ['AUTO', 'CPU', 'GPU', 'MYRIAD']
                device = st.selectbox(
                    "OpenVINO Device",
                    device_options,
                    index=device_options.index(self.config['detection']['device']),
                    help="Select OpenVINO inference device"
                )
                
                enable_int8 = st.checkbox(
                    "Enable INT8 Quantization",
                    value=self.config['detection']['enable_int8'],
                    help="Enable INT8 quantization for better performance"
                )
                
                async_inference = st.checkbox(
                    "Asynchronous Inference",
                    value=self.config['detection']['async_inference'],
                    help="Enable async inference for better performance"
                )
                
                # Show performance stats if available
                if hasattr(self.detector, 'get_performance_stats'):
                    stats = self.detector.get_performance_stats()
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("FPS", f"{stats.get('fps', 0):.1f}")
                        st.metric("Avg Time", f"{stats.get('avg_inference_time', 0)*1000:.1f}ms")
                    with col2:
                        st.metric("Frames", stats.get('frames_processed', 0))
                        st.metric("Backend", stats.get('backend', 'Unknown'))
            
            # Detection Settings
            with st.expander("🔍 Detection Settings", expanded=True):
                confidence_threshold = st.slider(
                    "Confidence Threshold",
                    min_value=0.1,
                    max_value=1.0,
                    value=self.config['detection']['confidence_threshold'],
                    step=0.05,
                    help="Minimum confidence for detections"
                )
                
                enable_ocr = st.checkbox(
                    "Enable License Plate OCR",
                    value=self.config['detection']['enable_ocr'],
                    help="Enable license plate recognition"
                )
                
                enable_tracking = st.checkbox(
                    "Enable Vehicle Tracking",
                    value=self.config['detection']['enable_tracking'],
                    help="Enable vehicle tracking for violation detection"
                )
            
            # Violation Settings
            with st.expander("🚨 Violation Detection", expanded=False):
                red_light_grace = st.number_input(
                    "Red Light Grace Period (seconds)",
                    min_value=0.5,
                    max_value=5.0,
                    value=self.config['violations']['red_light_grace_period'],
                    step=0.5
                )
                
                stop_duration = st.number_input(
                    "Required Stop Duration (seconds)",
                    min_value=1.0,
                    max_value=5.0,
                    value=self.config['violations']['stop_sign_duration'],
                    step=0.5
                )
                
                speed_tolerance = st.number_input(
                    "Speed Tolerance (km/h)",
                    min_value=0,
                    max_value=20,
                    value=self.config['violations']['speed_tolerance'],
                    step=1
                )
            
            # Display Settings
            with st.expander("🎨 Display Options", expanded=False):
                show_confidence = st.checkbox(
                    "Show Confidence Scores",
                    value=self.config['display']['show_confidence']
                )
                
                show_labels = st.checkbox(
                    "Show Detection Labels",
                    value=self.config['display']['show_labels']
                )
                
                show_license_plates = st.checkbox(
                    "Show License Plates",
                    value=self.config['display']['show_license_plates']
                )
                
                show_performance = st.checkbox(
                    "Show Performance Metrics",
                    value=self.config['display']['show_performance']
                )
            
            # Update configuration
            self.config.update({
                'detection': {
                    'confidence_threshold': confidence_threshold,
                    'enable_ocr': enable_ocr,
                    'enable_tracking': enable_tracking,
                    'device': device,
                    'enable_int8': enable_int8,
                    'async_inference': async_inference
                },
                'violations': {
                    'red_light_grace_period': red_light_grace,
                    'stop_sign_duration': stop_duration,
                    'speed_tolerance': speed_tolerance,
                    'enable_tracking': enable_tracking
                },
                'display': {
                    'show_confidence': show_confidence,
                    'show_labels': show_labels,
                    'show_license_plates': show_license_plates,
                    'show_performance': show_performance,
                    'max_display_width': 800
                }
            })
            
            # Control buttons
            st.divider()
            if st.button("🔄 Reload Models"):
                st.cache_resource.clear()
                st.rerun()
            
            if st.button("🗑️ Clear Data"):
                self._clear_all_data()
                st.success("Data cleared!")
    
    def _render_main_content(self):
        """Render main content area with tabs"""
        tab1, tab2, tab3, tab4 = st.tabs([
            "📹 Live Detection", 
            "📊 Analytics", 
            "🚨 Violations", 
            "📁 Export"
        ])
        
        with tab1:
            self._render_detection_tab()
        
        with tab2:
            self._render_analytics_tab()
        
        with tab3:
            self._render_violations_tab()
        
        with tab4:
            self._render_export_tab()
    
    def _render_detection_tab(self):
        """Render live detection tab"""
        st.header("📹 Live Traffic Detection")
        
        # Performance metrics display
        if self.config['display']['show_performance']:
            self._display_performance_metrics()
        
        # Input source selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            input_source = st.radio(
                "Select Input Source",
                ["Upload Video", "Webcam Stream", "Upload Image"],
                horizontal=True
            )
        
        with col2:
            if st.button("🔄 Reset Detection"):
                self._reset_detection()
        
        # Handle different input sources
        if input_source == "Upload Video":
            self._handle_video_upload()
        elif input_source == "Webcam Stream":
            self._handle_webcam_stream()
        else:  # Upload Image
            self._handle_image_upload()
    
    def _display_performance_metrics(self):
        """Display real-time performance metrics"""
        if hasattr(self.detector, 'get_performance_stats'):
            stats = self.detector.get_performance_stats()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "🚀 FPS",
                    f"{stats.get('fps', 0):.2f}",
                    delta=f"vs {stats.get('target_fps', 30):.0f} target"
                )
            
            with col2:
                avg_time_ms = stats.get('avg_inference_time', 0) * 1000
                st.metric(
                    "⚡ Avg Inference",
                    f"{avg_time_ms:.1f}ms",
                    delta=f"Backend: {stats.get('backend', 'Unknown')}"
                )
            
            with col3:
                st.metric(
                    "📊 Frames Processed",
                    stats.get('frames_processed', 0),
                    delta=f"Total detections: {stats.get('total_detections', 0)}"
                )
            
            with col4:
                # Performance indicator
                fps = stats.get('fps', 0)
                if fps > 25:
                    performance_status = "🟢 Excellent"
                    performance_color = "success"
                elif fps > 15:
                    performance_status = "🟡 Good"
                    performance_color = "warning"
                else:
                    performance_status = "🔴 Needs Optimization"
                    performance_color = "error"
                
                st.metric("📈 Performance", performance_status)
                
                # Show optimization suggestions
                if fps < 15:
                    st.info("💡 Try enabling INT8 quantization or changing device to GPU")
    
    def _handle_video_upload(self):
        """Handle video file upload and processing"""
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file for traffic analysis"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            import uuid
            unique_id = str(uuid.uuid4())[:8]
            tmp_path = os.path.join(tempfile.gettempdir(), f"traffic_video_{unique_id}.mp4")
            
            try:
                with open(tmp_path, 'wb') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                
                self._process_video_file(tmp_path)
                
            except Exception as e:
                st.error(f"Error processing video: {e}")
            finally:
                # Cleanup
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except:
                        pass
    
    def _process_video_file(self, video_path: str):
        """Process uploaded video file with OpenVINO acceleration"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("Error opening video file")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        st.info(f"📹 Video: {total_frames} frames at {fps:.1f} FPS")
        
        # Processing controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            frame_step = st.number_input(
                "Frame Step",
                min_value=1,
                max_value=10,
                value=1,
                help="Process every Nth frame"
            )
        
        with col2:
            max_frames = st.number_input(
                "Max Frames",
                min_value=10,
                max_value=min(total_frames, 1000),
                value=min(100, total_frames),
                help="Maximum frames to process"
            )
        
        with col3:
            if st.button("▶️ Process Video"):
                self._process_video_with_progress(cap, frame_step, max_frames)
        
        cap.release()
    def _process_video_with_progress(self, cap, frame_step: int, max_frames: int):
        """Process video with progress bar"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        frame_placeholder = st.empty()
        results_placeholder = st.empty()
        
        frame_count = 0
        processed_count = 0
        total_detections = 0
        total_violations = 0
        
        start_time = time.time()
        
        while cap.isOpened() and processed_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames based on frame_step
            if frame_count % frame_step == 0:
                # Process frame with detection
                try:
                    # Get detections using OpenVINO detector
                    detections = self.detector.detect_vehicles(
                        frame, 
                        conf_threshold=self.config['detection']['confidence_threshold']
                    )
                      # Process violations
                    violations = []
                    if self.violation_detector and detections:
                        violations = self.violation_detector.detect_violations(
                            detections, frame, frame_count
                        )
                      # Debug: Print detection format before annotation
                    self._debug_detection_format(detections, max_prints=2)
                    
                    # Draw detections and violations on frame
                    annotated_frame = self._annotate_frame(frame, detections, violations)
                    
                    # Update counters
                    frame_detections = len(detections) if detections else 0
                    frame_violations = len(violations) if violations else 0
                    total_detections += frame_detections
                    total_violations += frame_violations
                    
                    # Update session state
                    st.session_state.detection_count = total_detections
                    st.session_state.violation_count = total_violations
                    
                    # Store detection history
                    if detections:
                        for detection in detections:
                            detection['frame_number'] = processed_count
                            detection['timestamp'] = time.time()
                            self.detection_history.append(detection)
                    
                    # Store violation history
                    if violations:
                        for violation in violations:
                            violation['frame_number'] = processed_count
                            violation['timestamp'] = time.time()
                            self.violation_history.append(violation)
                    
                    # Update display
                    processed_count += 1
                    progress = processed_count / max_frames
                    progress_bar.progress(progress)
                    
                    # Update status
                    elapsed_time = time.time() - start_time
                    fps = processed_count / elapsed_time if elapsed_time > 0 else 0
                    
                    status_text.text(
                        f"Processing frame {processed_count}/{max_frames} "
                        f"({fps:.1f} FPS, {frame_detections} detections, {frame_violations} violations)"
                    )
                    
                    # Display frame
                    frame_placeholder.image(
                        cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                        caption=f"Frame {processed_count}",
                        use_container_width=True
                    )
                    
                    # Display results
                    with results_placeholder.container():
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("🚗 Detections", frame_detections)
                        with col2:
                            st.metric("🚨 Violations", frame_violations)
                    
                except Exception as e:
                    st.error(f"Error processing frame {processed_count}: {e}")
                    processed_count += 1
                    continue
            
            frame_count += 1
        
        # Final summary
        st.success(f"✅ Video processing complete! Processed {processed_count} frames")
        st.info(f"📊 Total Results: {total_detections} detections, {total_violations} violations")
        detections = self.detector.detect_vehicles(
                    frame,
                    conf_threshold=self.config['detection']['confidence_threshold']
                )
                
                # Detect violations
        violations = []
        if self.violation_detector and self.config['violations']['enable_tracking']:
                    violations = self.violation_detector.detect_violations(
                        detections, frame, time.time()
                    )
                
                # Annotate frame
        annotated_frame = self._annotate_frame(frame, detections, violations)
                
                # Display current frame
        with frame_placeholder.container():
                    st.image(
                        cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                        caption=f"Frame {frame_count}",
                        use_container_width=True
                    )
                
                # Update results
        with results_placeholder.container():
                    self._display_detection_results(detections, violations)
                
                # Store results
        self.detection_history.append(detections)
        self.violation_history.extend(violations)
                
        processed_count += 1
            
        frame_count += 1
            
            # Update progress
        progress = min(processed_count / max_frames, 1.0)
        progress_bar.progress(progress)
            
            # Update status
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
                fps = processed_count / elapsed_time
                status_text.text(
                    f"Processing frame {frame_count}: {processed_count}/{max_frames} "
                    f"({fps:.1f} FPS, {len(violations)} violations)"
                )
        
        st.success(f"✅ Video processing complete! Processed {processed_count} frames")
    
    def _handle_webcam_stream(self):
        """Handle webcam stream processing"""
        st.info("🎥 Webcam stream mode")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            start_webcam = st.button("▶️ Start Webcam", disabled=self.is_running)
        
        with col2:
            stop_webcam = st.button("⏸️ Stop Webcam", disabled=not self.is_running)
        
        with col3:
            capture_frame = st.button("📸 Capture Frame")
        
        if start_webcam:
            self._start_webcam_processing()
        
        if stop_webcam:
            self._stop_webcam_processing()
        
        if capture_frame and self.is_running:
            self._capture_current_frame()
        
        # Display webcam feed
        if self.is_running:
            self._display_webcam_feed()
    
    def _start_webcam_processing(self):
        """Start webcam processing"""
        try:
            self.cap = cv2.VideoCapture(0)
            self.is_running = True
            st.success("✅ Webcam started")
        except Exception as e:
            st.error(f"Error starting webcam: {e}")
    
    def _stop_webcam_processing(self):
        """Stop webcam processing"""
        if hasattr(self, 'cap'):
            self.cap.release()
        self.is_running = False
        st.success("⏸️ Webcam stopped")
    
    def _display_webcam_feed(self):
        """Display live webcam feed with detection"""
        if not hasattr(self, 'cap') or not self.cap.isOpened():
            return
        
        webcam_placeholder = st.empty()
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                st.error("Failed to read from webcam")
                break
            
            # Process frame
            start_time = time.time()
            detections = self.detector.detect_vehicles(
                frame,
                conf_threshold=self.config['detection']['confidence_threshold']
            )
            processing_time = time.time() - start_time
            
            # Detect violations
            violations = []
            if self.violation_detector and self.config['violations']['enable_tracking']:
                violations = self.violation_detector.detect_violations(
                    detections, frame, time.time()
                )
            
            # Annotate frame
            annotated_frame = self._annotate_frame(frame, detections, violations)
            
            # Display frame
            with webcam_placeholder.container():
                st.image(
                    cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                    caption=f"Live Webcam Feed - Processing: {processing_time*1000:.1f}ms",
                    use_container_width=True
                )
            
            # Update history
            self.detection_history.append(detections)
            self.violation_history.extend(violations)
            st.session_state.processed_frames += 1
            
            # Break loop if not running
            if not self.is_running:
                break
            
            # Small delay for UI responsiveness
            time.sleep(0.1)
    
    def _handle_image_upload(self):
        """Handle single image upload and processing"""
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image for traffic analysis"
        )
        
        if uploaded_file is not None:
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            
            # Display original image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📸 Original Image")
                st.image(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    use_container_width=True
                )
            
            # Process image
            with st.spinner("Processing image..."):
                detections = self.detector.detect_vehicles(
                    image,
                    conf_threshold=self.config['detection']['confidence_threshold']
                )
                
                # Detect violations (static analysis)
                violations = []
                if self.violation_detector:
                    violations = self.violation_detector.detect_violations(
                        detections, image, time.time()
                    )
                
                # Annotate image
                annotated_image = self._annotate_frame(image, detections, violations)
            
            with col2:
                st.subheader("🔍 Detected Results")
                st.image(
                    cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB),
                    use_container_width=True
                )
              # Display results
            self._display_detection_results(detections, violations)
    
    def _debug_detection_format(self, detections, max_prints=3):
        """Debug function to print detection format and structure"""
        if detections is None:
            print("DEBUG: detections is None")
            return
            
        print(f"DEBUG: detections type: {type(detections)}")
        print(f"DEBUG: detections length: {len(detections)}")
        
        if len(detections) > 0:
            for i, det in enumerate(detections[:max_prints]):
                print(f"DEBUG: Detection {i}:")
                print(f"  Type: {type(det)}")
                if isinstance(det, dict):
                    print(f"  Keys: {list(det.keys())}")
                    print(f"  bbox: {det.get('bbox', 'MISSING')}")
                    print(f"  confidence: {det.get('confidence', 'MISSING')}")
                    print(f"  class_name: {det.get('class_name', 'MISSING')}")
                elif isinstance(det, np.ndarray):
                    print(f"  Shape: {det.shape}")
                    print(f"  Dtype: {det.dtype}")
                    if hasattr(det, 'dtype') and det.dtype.names:
                        print(f"  Field names: {det.dtype.names}")
                else:
                    print(f"  Value: {det}")

    def _convert_detections_to_dict(self, detections):
        """Convert numpy structured arrays to dictionary format for annotation"""
        if detections is None:
            return []
            
        converted_detections = []
        
        for det in detections:
            try:
                if isinstance(det, dict):
                    # Already in correct format
                    converted_detections.append(det)
                elif isinstance(det, np.ndarray) and det.dtype.names:
                    # Structured numpy array - convert to dict
                    det_dict = {}
                    for field in det.dtype.names:
                        value = det[field]
                        # Handle numpy types
                        if isinstance(value, np.ndarray):
                            det_dict[field] = value.tolist()
                        elif isinstance(value, (np.integer, np.floating)):
                            det_dict[field] = float(value)
                        else:
                            det_dict[field] = value
                    converted_detections.append(det_dict)
                elif isinstance(det, (list, tuple)) and len(det) >= 6:
                    # Legacy format [x1, y1, x2, y2, confidence, class_id]
                    # Use traffic class names list
                    traffic_class_names = [
                        'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
                        'traffic light', 'stop sign', 'parking meter'
                    ]
                    class_id = int(det[5])
                    class_name = traffic_class_names[class_id] if class_id < len(traffic_class_names) else 'unknown'
                    det_dict = {
                        'bbox': list(det[:4]),
                        'confidence': float(det[4]),
                        'class_id': class_id,
                        'class_name': class_name
                    }
                    converted_detections.append(det_dict)
                else:
                    print(f"Warning: Unknown detection format: {type(det)}")
                    continue
            except Exception as e:
                print(f"Error converting detection: {e}")
                continue
                
        return converted_detections

    def _validate_and_fix_bbox(self, bbox, frame_width, frame_height):
        """Validate and fix bounding box coordinates"""
        try:
            if not bbox or len(bbox) < 4:
                return None
                
            # Convert to float first, then int
            x1, y1, x2, y2 = map(float, bbox[:4])
            
            # Check if coordinates are normalized (0-1 range)
            if all(0 <= coord <= 1 for coord in [x1, y1, x2, y2]):
                # Convert normalized coordinates to pixel coordinates
                x1 = int(x1 * frame_width)
                y1 = int(y1 * frame_height)
                x2 = int(x2 * frame_width)
                y2 = int(y2 * frame_height)
            else:
                # Assume already in pixel coordinates
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Ensure coordinates are within frame bounds
            x1 = max(0, min(x1, frame_width - 1))
            y1 = max(0, min(y1, frame_height - 1))
            x2 = max(0, min(x2, frame_width))
            y2 = max(0, min(y2, frame_height))
            
            # Ensure valid box dimensions
            if x2 <= x1:
                x2 = x1 + 1
            if y2 <= y1:
                y2 = y1 + 1
                
            return [x1, y1, x2, y2]
            
        except Exception as e:
            print(f"Error validating bbox {bbox}: {e}")
            return None

    def _annotate_frame(self, frame, detections, violations):
        """Draw bounding boxes and labels for detections on the frame."""
        import cv2
        import numpy as np
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]

        # Debug: Print the first detection
        if detections and len(detections) > 0:
            print('Sample detection:', detections[0])

        for det in detections or []:
            bbox = det.get('bbox')
            if bbox is None or len(bbox) < 4:
                continue
            # If coordinates are normalized (0-1), scale to pixel values
            if max(bbox) <= 1.0:
                x1 = int(bbox[0] * w)
                y1 = int(bbox[1] * h)
                x2 = int(bbox[2] * w)
                y2 = int(bbox[3] * h)
            else:
                x1, y1, x2, y2 = map(int, bbox[:4])
            # Ensure coordinates are valid
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w-1))
            y2 = max(0, min(y2, h-1))
            if x2 <= x1 or y2 <= y1:
                continue
            label = det.get('class_name') or det.get('label', 'object')
            confidence = det.get('confidence', 0.0)
            color = (0, 255, 0)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, f'{label} {confidence:.2f}', (x1, max(y1-10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return annotated_frame
    
    def _display_detection_results(self, detections: List[Dict], violations: List[Dict]):
        """Display detection and violation results"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚗 Detections")
            if detections:
                # Group by type
                detection_summary = {}
                for detection in detections:
                    det_type = detection.get('type', 'unknown')
                    detection_summary[det_type] = detection_summary.get(det_type, 0) + 1
                
                for det_type, count in detection_summary.items():
                    st.write(f"- {det_type.replace('_', ' ').title()}: {count}")
                
                # Show details in expander
                with st.expander("Detection Details"):
                    for i, detection in enumerate(detections):
                        st.write(f"{i+1}. {detection['class_name']} "
                                f"(conf: {detection['confidence']:.2f})")
                        if detection.get('license_plate'):
                            st.write(f"   License: {detection['license_plate']}")
            else:
                st.info("No detections found")
        
        with col2:
            st.subheader("🚨 Violations")
            if violations:
                for violation in violations:                    # Make sure violation is a dictionary
                    if not isinstance(violation, dict):
                        continue
                        
                    severity_color = {
                        'high': '🔴',
                        'medium': '🟡',
                        'low': '🟢'
                    }.get(violation.get('severity', 'medium'), '🔵')
                    
                    st.markdown(
                        f'<div class="violation-alert">'
                        f'{severity_color} <strong>{violation.get("type", "Unknown").replace("_", " ").title()}</strong><br/>'
                        f'{violation.get("description", "No description")}<br/>'
                        f'<small>Confidence: {violation.get("confidence", 0):.2f}</small>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.info("No violations detected")
    
    def _render_analytics_tab(self):
        """Render analytics dashboard"""
        st.header("📊 Traffic Analytics Dashboard")
        
        if not self.detection_history:
            st.info("No data available. Start processing videos or images to see analytics.")
            return
        
        # Overall statistics
        st.subheader("📈 Overall Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_detections = sum(len(frame_dets) for frame_dets in self.detection_history)
        total_violations = len(self.violation_history)
        avg_detections_per_frame = total_detections / len(self.detection_history) if self.detection_history else 0
        uptime = time.time() - st.session_state.start_time
        
        with col1:
            st.metric("Total Detections", total_detections)
        with col2:
            st.metric("Total Violations", total_violations)
        with col3:
            st.metric("Avg Detections/Frame", f"{avg_detections_per_frame:.1f}")
        with col4:
            st.metric("Uptime", f"{uptime/3600:.1f}h")
        
        # Detection trends
        if len(self.detection_history) > 10:
            st.subheader("📊 Detection Trends")
            
            detection_counts = [len(frame_dets) for frame_dets in self.detection_history[-50:]]
            df_trend = pd.DataFrame({
                'Frame': range(len(detection_counts)),
                'Detections': detection_counts
            })
            
            st.line_chart(df_trend.set_index('Frame'))
        
        # Vehicle type distribution
        st.subheader("🚗 Vehicle Type Distribution")
        vehicle_types = {}
        
        for frame_detections in self.detection_history:
            for detection in frame_detections:
                if detection.get('type') == 'vehicle':
                    vehicle_type = detection.get('vehicle_type', 'unknown')
                    vehicle_types[vehicle_type] = vehicle_types.get(vehicle_type, 0) + 1
        
        if vehicle_types:
            df_vehicles = pd.DataFrame(
                list(vehicle_types.items()),
                columns=['Vehicle Type', 'Count']
            )
            st.bar_chart(df_vehicles.set_index('Vehicle Type'))
        
        # Violation analysis
        if self.violation_history:
            st.subheader("🚨 Violation Analysis")
            
            violation_types = {}
            for violation in self.violation_history:
                v_type = violation['type']
                violation_types[v_type] = violation_types.get(v_type, 0) + 1
            
            df_violations = pd.DataFrame(
                list(violation_types.items()),
                columns=['Violation Type', 'Count']
            )
            st.bar_chart(df_violations.set_index('Violation Type'))
        
        # Performance analytics
        if hasattr(self.detector, 'get_performance_stats'):
            st.subheader("⚡ Performance Analytics")
            stats = self.detector.get_performance_stats()
            
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            
            with perf_col1:
                st.metric("Average FPS", f"{stats.get('fps', 0):.2f}")
                st.metric("Total Frames", stats.get('frames_processed', 0))
            
            with perf_col2:
                st.metric("Avg Inference Time", f"{stats.get('avg_inference_time', 0)*1000:.1f}ms")
                st.metric("Backend Used", stats.get('backend', 'Unknown'))
            
            with perf_col3:
                st.metric("Total Detections", stats.get('total_detections', 0))
                st.metric("Detection Rate", f"{stats.get('detection_rate', 0):.1f}/frame")
    
    def _render_violations_tab(self):
        """Render violations monitoring tab"""
        st.header("🚨 Traffic Violations Monitor")
        
        if not self.violation_history:
            st.info("No violations detected yet. Start processing videos or streams to monitor violations.")
            return
        
        # Violation statistics
        st.subheader("📊 Violation Statistics")
        
        violation_summary = {}
        severity_summary = {'high': 0, 'medium': 0, 'low': 0}
        for violation in self.violation_history:
            # Make sure violation is a dictionary
            if not isinstance(violation, dict):
                continue
                
            v_type = violation.get('type', 'unknown')
            severity = violation.get('severity', 'medium')
            
            violation_summary[v_type] = violation_summary.get(v_type, 0) + 1
            severity_summary[severity] += 1
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**By Type:**")
            for v_type, count in violation_summary.items():
                st.write(f"- {v_type.replace('_', ' ').title()}: {count}")
        
        with col2:
            st.write("**By Severity:**")
            for severity, count in severity_summary.items():
                color = {"high": "🔴", "medium": "🟡", "low": "🟢"}[severity]
                st.write(f"- {color} {severity.title()}: {count}")
          # Recent violations
        st.subheader("🕐 Recent Violations")
        
        recent_violations = self.violation_history[-10:]  # Last 10 violations
        for i, violation in enumerate(reversed(recent_violations), 1):
            # Make sure violation is a dictionary
            if not isinstance(violation, dict):
                continue
                
            timestamp = violation.get('timestamp', time.time())
            time_str = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
            
            severity_icon = {
                'high': '🔴',
                'medium': '🟡', 
                'low': '🟢'
            }.get(violation.get('severity', 'medium'), '🔵')
            
            st.markdown(
                f'<div class="violation-alert">'
                f'<strong>{i}. {severity_icon} {violation.get("type", "Unknown").replace("_", " ").title()}</strong> '
                f'<small>({time_str})</small><br/>'
                f'{violation["description"]}<br/>'
                f'<small>Confidence: {violation.get("confidence", 0):.2f} | '
                f'Severity: {violation.get("severity", "medium").title()}</small>'
                f'</div>',
                unsafe_allow_html=True
            )
        
        # Violation trends
        if len(self.violation_history) > 5:
            st.subheader("📈 Violation Trends")
            
            # Group violations by hour
            violation_times = [v.get('timestamp', time.time()) for v in self.violation_history]
            violation_hours = [datetime.fromtimestamp(t).hour for t in violation_times]
            
            hour_counts = {}
            for hour in violation_hours:
                hour_counts[hour] = hour_counts.get(hour, 0) + 1
            
            df_hourly = pd.DataFrame(
                list(hour_counts.items()),
                columns=['Hour', 'Violations']
            )
            
            st.bar_chart(df_hourly.set_index('Hour'))
    
    def _render_export_tab(self):
        """Render data export tab"""
        st.header("📁 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Detection Data")
            
            if self.detection_history:
                # Generate CSV data for detections
                detection_data = []
                for frame_idx, frame_detections in enumerate(self.detection_history):
                    for detection in frame_detections:
                        detection_data.append({
                            'frame_id': frame_idx,
                            'timestamp': datetime.now().isoformat(),
                            'class_name': detection['class_name'],
                            'confidence': detection['confidence'],
                            'bbox_x1': detection['bbox'][0],
                            'bbox_y1': detection['bbox'][1],
                            'bbox_x2': detection['bbox'][2],
                            'bbox_y2': detection['bbox'][3],
                            'type': detection.get('type', 'unknown'),
                            'vehicle_type': detection.get('vehicle_type', ''),
                            'license_plate': detection.get('license_plate', '')
                        })
                
                df_detections = pd.DataFrame(detection_data)
                csv_detections = df_detections.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Detection Data (CSV)",
                    data=csv_detections,
                    file_name=f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                st.write(f"**Total Records:** {len(detection_data)}")
            else:
                st.info("No detection data available")
        
        with col2:
            st.subheader("🚨 Violation Data")
            
            if self.violation_history:
                # Generate CSV data for violations
                violation_data = []
                for violation in self.violation_history:
                    violation_data.append({
                        'timestamp': datetime.fromtimestamp(violation.get('timestamp', time.time())).isoformat(),
                        'type': violation['type'],
                        'description': violation['description'],
                        'severity': violation.get('severity', 'medium'),
                        'confidence': violation.get('confidence', 0),
                        'vehicle_id': violation.get('vehicle_id', ''),
                        'location': violation.get('location', ''),
                        'bbox_x1': violation.get('bbox', [0,0,0,0])[0],
                        'bbox_y1': violation.get('bbox', [0,0,0,0])[1],
                        'bbox_x2': violation.get('bbox', [0,0,0,0])[2],
                        'bbox_y2': violation.get('bbox', [0,0,0,0])[3]
                    })
                
                df_violations = pd.DataFrame(violation_data)
                csv_violations = df_violations.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Violation Data (CSV)",
                    data=csv_violations,
                    file_name=f"violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                st.write(f"**Total Violations:** {len(violation_data)}")
            else:
                st.info("No violation data available")
        
        # Export configuration
        st.subheader("⚙️ Configuration Export")
        
        config_json = json.dumps(self.config, indent=2)
        st.download_button(
            label="📥 Download Configuration (JSON)",
            data=config_json,
            file_name=f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        # Performance report
        if hasattr(self.detector, 'get_performance_stats'):
            st.subheader("📈 Performance Report")
            
            stats = self.detector.get_performance_stats()
            performance_report = json.dumps(stats, indent=2)
            
            st.download_button(
                label="📥 Download Performance Report (JSON)",
                data=performance_report,
                file_name=f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    def _reset_detection(self):
        """Reset all detection data"""
        self.detection_history = []
        self.violation_history = []
        st.session_state.detection_count = 0
        st.session_state.violation_count = 0
        st.session_state.processed_frames = 0
        st.success("Detection data reset successfully!")
    
    def _clear_all_data(self):
        """Clear all application data"""
        self._reset_detection()
        if hasattr(self.detector, 'reset_performance_stats'):
            self.detector.reset_performance_stats()
        st.session_state.performance_stats = {}
    def _capture_current_frame(self):
        """Capture and save current frame"""
        if hasattr(self, 'cap') and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"captured_frame_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                st.success(f"📸 Frame captured: {filename}")

    def _debug_detection_format(self, detections, max_prints=3):
        """Debug function to print detection format and structure"""
        if detections is None:
            print("DEBUG: detections is None")
            return
            
        print(f"DEBUG: detections type: {type(detections)}")
        print(f"DEBUG: detections length: {len(detections)}")
        
        if len(detections) > 0:
            for i, det in enumerate(detections[:max_prints]):
                print(f"DEBUG: Detection {i}:")
                print(f"  Type: {type(det)}")
                if isinstance(det, dict):
                    print(f"  Keys: {list(det.keys())}")
                    print(f"  bbox: {det.get('bbox', 'MISSING')}")
                    print(f"  confidence: {det.get('confidence', 'MISSING')}")
                    print(f"  class_name: {det.get('class_name', 'MISSING')}")
                elif isinstance(det, np.ndarray):
                    print(f"  Shape: {det.shape}")
                    print(f"  Dtype: {det.dtype}")
                    if hasattr(det, 'dtype') and det.dtype.names:
                        print(f"  Field names: {det.dtype.names}")
                else:
                    print(f"  Value: {det}")

    def _convert_detections_to_dict(self, detections):
        """Convert numpy structured arrays to dictionary format for annotation"""
        if detections is None:
            return []
            
        converted_detections = []
        
        for det in detections:
            try:
                if isinstance(det, dict):
                    # Already in correct format
                    converted_detections.append(det)
                elif isinstance(det, np.ndarray) and det.dtype.names:
                    # Structured numpy array - convert to dict
                    det_dict = {}
                    for field in det.dtype.names:
                        value = det[field]
                        # Handle numpy types
                        if isinstance(value, np.ndarray):
                            det_dict[field] = value.tolist()
                        elif isinstance(value, (np.integer, np.floating)):
                            det_dict[field] = float(value)
                        else:
                            det_dict[field] = value
                    converted_detections.append(det_dict)
                elif isinstance(det, (list, tuple)) and len(det) >= 6:
                    # Legacy format [x1, y1, x2, y2, confidence, class_id]
                    # Use traffic class names list
                    traffic_class_names = [
                        'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
                        'traffic light', 'stop sign', 'parking meter'
                    ]
                    class_id = int(det[5])
                    class_name = traffic_class_names[class_id] if class_id < len(traffic_class_names) else 'unknown'
                    det_dict = {
                        'bbox': list(det[:4]),
                        'confidence': float(det[4]),
                        'class_id': class_id,
                        'class_name': class_name
                    }
                    converted_detections.append(det_dict)
                else:
                    print(f"Warning: Unknown detection format: {type(det)}")
                    continue
            except Exception as e:
                print(f"Error converting detection: {e}")
                continue
                
        return converted_detections

    def _validate_and_fix_bbox(self, bbox, frame_width, frame_height):
        """Validate and fix bounding box coordinates"""
        try:
            if not bbox or len(bbox) < 4:
                return None
                
            # Convert to float first, then int
            x1, y1, x2, y2 = map(float, bbox[:4])
            
            # Check if coordinates are normalized (0-1 range)
            if all(0 <= coord <= 1 for coord in [x1, y1, x2, y2]):
                # Convert normalized coordinates to pixel coordinates
                x1 = int(x1 * frame_width)
                y1 = int(y1 * frame_height)
                x2 = int(x2 * frame_width)
                y2 = int(y2 * frame_height)
            else:
                # Assume already in pixel coordinates
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Ensure coordinates are within frame bounds
            x1 = max(0, min(x1, frame_width - 1))
            y1 = max(0, min(y1, frame_height - 1))
            x2 = max(0, min(x2, frame_width))
            y2 = max(0, min(y2, frame_height))
            
            # Ensure valid box dimensions
            if x2 <= x1:
                x2 = x1 + 1
            if y2 <= y1:
                y2 = y1 + 1
                
            return [x1, y1, x2, y2]
            
        except Exception as e:
            print(f"Error validating bbox {bbox}: {e}")
            return None

    # ...existing code...
def main():
    """Main application entry point"""
    try:
        app = TrafficMonitoringApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.error("Please check that all required modules are properly installed.")

if __name__ == "__main__":
    main()