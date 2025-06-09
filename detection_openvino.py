# Detection logic using OpenVINO models (YOLO, etc.)

import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# --- Install required packages if missing ---
try:
    import openvino as ov
except ImportError:
    print("Installing openvino...")
    os.system('pip install --quiet "openvino>=2024.0.0"')
    import openvino as ov
try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    os.system('pip install --quiet "ultralytics==8.3.0"')
    from ultralytics import YOLO
try:
    import nncf
except ImportError:
    print("Installing nncf...")
    os.system('pip install --quiet "nncf>=2.9.0"')
    import nncf

# --- Traffic-related class names ---
TRAFFIC_CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
    'traffic light', 'stop sign', 'parking meter'
]

# --- Model Conversion and Quantization ---
def convert_yolo_to_openvino(model_name: str = "yolo11x", half: bool = True) -> Path:
    """Convert YOLOv11x PyTorch model to OpenVINO IR format."""
    pt_path = Path(f"{model_name}.pt")
    ov_dir = Path(f"{model_name}_openvino_model")
    ov_xml = ov_dir / f"{model_name}.xml"
    if not ov_xml.exists():
        print(f"Exporting {pt_path} to OpenVINO IR...")
        model = YOLO(str(pt_path))
        model.export(format="openvino", dynamic=True, half=half)
    else:
        print(f"OpenVINO IR already exists: {ov_xml}")
    return ov_xml

def quantize_openvino_model(ov_xml: Path, model_name: str = "yolo11x") -> Path:
    """Quantize OpenVINO IR model to INT8 using NNCF."""
    int8_dir = Path(f"{model_name}_openvino_int8_model")
    int8_xml = int8_dir / f"{model_name}.xml"
    if int8_xml.exists():
        print(f"INT8 model already exists: {int8_xml}")
        return int8_xml
    print("Quantization requires a calibration dataset. Skipping actual quantization in this demo.")
    return ov_xml  # Return FP32 if no quantization

# --- OpenVINO Inference Pipeline ---
class OpenVINOYOLODetector:
    def __init__(self, model_xml: Path, device: str = "AUTO"):
        self.core = ov.Core()
        self.device = device
        self.model = self.core.read_model(model_xml)
        self.input_shape = self.model.inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        self.ov_config = {}
        if device != "CPU":
            self.model.reshape({0: [1, 3, 640, 640]})
        if "GPU" in device or ("AUTO" in device and "GPU" in self.core.available_devices):
            self.ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
        self.compiled_model = self.core.compile_model(self.model, device, self.ov_config)
        self.output_layer = self.compiled_model.output(0)

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame, (self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[None]
        return img

    def infer(self, frame: np.ndarray, conf_threshold: float = 0.25) -> List[Dict]:
        input_tensor = self.preprocess(frame)
        output = self.compiled_model([input_tensor])[self.output_layer]
        return self.postprocess(output, frame.shape, conf_threshold)

    def postprocess(self, output: np.ndarray, frame_shape, conf_threshold: float) -> List[Dict]:
        # Output: (1, 84, 8400) or (84, 8400) or (8400, 84)
        if output.ndim == 3:
            output = np.squeeze(output)
        if output.shape[0] == 84:
            output = output.T  # (8400, 84)
        boxes = output[:, :4]
        scores = output[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        detections = []
        h, w = frame_shape[:2]
        for i, (box, score, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            if score < conf_threshold:
                continue
            x_c, y_c, bw, bh = box
            # If normalized, scale to input size
            if all(0.0 <= v <= 1.0 for v in box):
                x_c *= self.input_width
                y_c *= self.input_height
                bw *= self.input_width
                bh *= self.input_height
            # Scale to original frame size
            scale_x = w / self.input_width
            scale_y = h / self.input_height
            x_c *= scale_x
            y_c *= scale_y
            bw *= scale_x
            bh *= scale_y
            x1 = int(round(x_c - bw / 2))
            y1 = int(round(y_c - bh / 2))
            x2 = int(round(x_c + bw / 2))
            y2 = int(round(y_c + bh / 2))
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            class_name = TRAFFIC_CLASS_NAMES[class_id] if class_id < len(TRAFFIC_CLASS_NAMES) else str(class_id)
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(score),
                'class_id': int(class_id),
                'class_name': class_name
            })
        return detections

    def draw(self, frame: np.ndarray, detections: List[Dict], box_thickness: int = 2) -> np.ndarray:
        # 80+ visually distinct colors for COCO classes (BGR)
        COCO_COLORS = [
            (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29), (207, 210, 49),
            (72, 249, 10), (146, 204, 23), (61, 219, 134), (26, 147, 52), (0, 212, 187),
            (44, 153, 168), (0, 194, 255), (52, 69, 147), (100, 115, 255), (0, 24, 236),
            (132, 56, 255), (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
            (255, 255, 56), (255, 255, 151), (255, 255, 31), (255, 255, 29), (207, 255, 49),
            (72, 255, 10), (146, 255, 23), (61, 255, 134), (26, 255, 52), (0, 255, 187),
            (44, 255, 168), (0, 255, 255), (52, 255, 147), (100, 255, 255), (0, 255, 236),
            (132, 255, 255), (82, 255, 133), (203, 255, 255), (255, 255, 200), (255, 255, 199),
            (56, 255, 255), (157, 255, 151), (112, 255, 31), (178, 255, 29), (210, 255, 49),
            (249, 255, 10), (204, 255, 23), (219, 255, 134), (147, 255, 52), (212, 255, 187),
            (153, 255, 168), (194, 255, 255), (69, 255, 147), (115, 255, 255), (24, 255, 236),
            (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49),
            (249, 72, 10), (204, 146, 23), (219, 61, 134), (147, 26, 52), (212, 0, 187),
            (153, 44, 168), (194, 0, 255), (69, 52, 147), (115, 100, 255), (24, 0, 236),
            (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49),
            (249, 72, 10), (204, 146, 23), (219, 61, 134), (147, 26, 52), (212, 0, 187),
            (153, 44, 168), (194, 0, 255), (69, 52, 147), (115, 100, 255), (24, 0, 236),
            (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49)
        ]
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class_name']} {det['confidence']:.2f}"
            color = COCO_COLORS[det['class_id'] % len(COCO_COLORS)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
            cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

# --- Video/Image/Live Inference ---
def run_inference(detector: OpenVINOYOLODetector, source=0, conf_threshold=0.25, flip=False, use_popup=False, video_width=None):
    if isinstance(source, str) and not os.path.exists(source):
        print(f"Downloading sample video: {source}")
        import requests
        url = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/people.mp4"
        r = requests.get(url)
        with open(source, 'wb') as f:
            f.write(r.content)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Failed to open video source: {source}")
        return
    window_name = "YOLOv11x + OpenVINO Detection"
    if use_popup:
        cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
    frame_count = 0
    times = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if flip:
            frame = cv2.flip(frame, 1)
        if video_width:
            scale = video_width / max(frame.shape[:2])
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        start = time.time()
        detections = detector.infer(frame, conf_threshold=conf_threshold)
        frame = detector.draw(frame, detections)
        elapsed = time.time() - start
        times.append(elapsed)
        if len(times) > 200:
            times.pop(0)
        fps = 1.0 / np.mean(times) if times else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        if use_popup:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()

# --- Main Entrypoint ---
if __name__ == "__main__":
    # Choose model: yolo11x or yolo11n, etc.
    MODEL_NAME = "yolo11x"
    DEVICE = "AUTO"  # or "CPU", "GPU"
    # Step 1: Convert model if needed
    ov_xml = convert_yolo_to_openvino(MODEL_NAME)
    # Step 2: Quantize (optional, demo skips actual quantization)
    ov_xml = quantize_openvino_model(ov_xml, MODEL_NAME)
    # Step 3: Create detector
    detector = OpenVINOYOLODetector(ov_xml, device=DEVICE)
    # Step 4: Run on webcam, video, or image
    # Webcam: source=0, Video: source="video.mp4", Image: source="image.jpg"
    run_inference(detector, source=0, conf_threshold=0.25, flip=True, use_popup=True, video_width=1280)
# To run on a video file: run_inference(detector, source="people.mp4", conf_threshold=0.25)
# To run on an image: run_inference(detector, source="image.jpg", conf_threshold=0.25)
# To run async or batch, extend the OpenVINOYOLODetector class with async API as needed.

import os
import time
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional

# Only traffic-related classes for detection
TRAFFIC_CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
    'traffic light', 'stop sign', 'parking meter'
]

class OpenVINOVehicleDetector:
    def __init__(self, model_path: str = None, device: str = "AUTO", use_quantized: bool = False, enable_ocr: bool = False, confidence_threshold: float = 0.4):
        import openvino as ov
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.ocr_reader = None
        self.class_names = TRAFFIC_CLASS_NAMES
        self.performance_stats = {
            'fps': 0,
            'avg_inference_time': 0,
            'frames_processed': 0,
            'backend': f"OpenVINO-{device}",
            'total_detections': 0,
            'detection_rate': 0
        }
        self._inference_times = []
        self._start_time = time.time()
        self._frame_count = 0
        # Model selection logic
        self.model_path = self._find_best_model(model_path, use_quantized)
        self.core = ov.Core()
        self.model = self.core.read_model(self.model_path)
        # Always reshape to static shape before accessing .shape
        self.model.reshape({0: [1, 3, 640, 640]})
        self.input_shape = self.model.inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        self.ov_config = {}
        if device != "CPU":
            # Already reshaped above, so nothing more needed here
            pass
        if "GPU" in device or ("AUTO" in device and "GPU" in self.core.available_devices):
            self.ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
        self.compiled_model = self.core.compile_model(self.model, device, self.ov_config)
        self.output_layer = self.compiled_model.output(0)

    def _find_best_model(self, model_path, use_quantized):
        # Priority: quantized IR > IR > .pt
        search_paths = [
            Path(model_path) if model_path else None,
            Path("yolo11x_openvino_int8_model/yolo11x.xml") if use_quantized else None,
            Path("yolo11x_openvino_model/yolo11x.xml"),
            Path("rcb/yolo11x_openvino_model/yolo11x.xml"),
            Path("yolo11x.xml"),
            Path("rcb/yolo11x.xml"),
            Path("yolo11x.pt"),
            Path("rcb/yolo11x.pt")
        ]
        for p in search_paths:
            if p and p.exists():
                return str(p)
        raise FileNotFoundError("No suitable YOLOv11x model found for OpenVINO.")

    def detect_vehicles(self, frame: np.ndarray, conf_threshold: float = None) -> List[Dict]:
        if conf_threshold is None:
            conf_threshold = self.confidence_threshold
        start = time.time()
        input_tensor = self._preprocess(frame)
        output = self.compiled_model([input_tensor])[self.output_layer]
        detections = self._postprocess(output, frame.shape, conf_threshold)
        elapsed = time.time() - start
        self._inference_times.append(elapsed)
        self._frame_count += 1
        self.performance_stats['frames_processed'] = self._frame_count
        self.performance_stats['total_detections'] += len(detections)
        if len(self._inference_times) > 100:
            self._inference_times.pop(0)
        self.performance_stats['avg_inference_time'] = float(np.mean(self._inference_times)) if self._inference_times else 0
        total_time = time.time() - self._start_time
        self.performance_stats['fps'] = self._frame_count / total_time if total_time > 0 else 0
        return detections

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame, (self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[None]
        return img

    def _postprocess(self, output: np.ndarray, frame_shape, conf_threshold: float) -> List[Dict]:
        # Output: (1, 84, 8400) or (84, 8400) or (8400, 84)
        if output.ndim == 3:
            output = np.squeeze(output)
        if output.shape[0] == 84:
            output = output.T  # (8400, 84)
        boxes = output[:, :4]
        scores = output[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        detections = []
        h, w = frame_shape[:2]
        for i, (box, score, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            if score < conf_threshold:
                continue
            x_c, y_c, bw, bh = box
            # If normalized, scale to input size
            if all(0.0 <= v <= 1.0 for v in box):
                x_c *= self.input_width
                y_c *= self.input_height
                bw *= self.input_width
                bh *= self.input_height
            # Scale to original frame size
            scale_x = w / self.input_width
            scale_y = h / self.input_height
            x_c *= scale_x
            y_c *= scale_y
            bw *= scale_x
            bh *= scale_y
            x1 = int(round(x_c - bw / 2))
            y1 = int(round(y_c - bh / 2))
            x2 = int(round(x_c + bw / 2))
            y2 = int(round(y_c + bh / 2))
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            class_name = TRAFFIC_CLASS_NAMES[class_id] if class_id < len(TRAFFIC_CLASS_NAMES) else str(class_id)
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(score),
                'class_id': int(class_id),
                'class_name': class_name
            })
        return detections

    def draw(self, frame: np.ndarray, detections: List[Dict], box_thickness: int = 2) -> np.ndarray:
        # 80+ visually distinct colors for COCO classes (BGR)
        COCO_COLORS = [
            (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29), (207, 210, 49),
            (72, 249, 10), (146, 204, 23), (61, 219, 134), (26, 147, 52), (0, 212, 187),
            (44, 153, 168), (0, 194, 255), (52, 69, 147), (100, 115, 255), (0, 24, 236),
            (132, 56, 255), (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
            (255, 255, 56), (255, 255, 151), (255, 255, 31), (255, 255, 29), (207, 255, 49),
            (72, 255, 10), (146, 255, 23), (61, 255, 134), (26, 255, 52), (0, 255, 187),
            (44, 255, 168), (0, 255, 255), (52, 255, 147), (100, 255, 255), (0, 255, 236),
            (132, 255, 255), (82, 255, 133), (203, 255, 255), (255, 255, 200), (255, 255, 199),
            (56, 255, 255), (157, 255, 151), (112, 255, 31), (178, 255, 29), (210, 255, 49),
            (249, 255, 10), (204, 255, 23), (219, 255, 134), (147, 255, 52), (212, 255, 187),
            (153, 255, 168), (194, 255, 255), (69, 255, 147), (115, 255, 255), (24, 255, 236),
            (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49),
            (249, 72, 10), (204, 146, 23), (219, 61, 134), (147, 26, 52), (212, 0, 187),
            (153, 44, 168), (194, 0, 255), (69, 52, 147), (115, 100, 255), (24, 0, 236),
            (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49),
            (249, 72, 10), (204, 146, 23), (219, 61, 134), (147, 26, 52), (212, 0, 187),
            (153, 44, 168), (194, 0, 255), (69, 52, 147), (115, 100, 255), (24, 0, 236),
            (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49)
        ]
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class_name']} {det['confidence']:.2f}"
            color = COCO_COLORS[det['class_id'] % len(COCO_COLORS)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
            cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

# --- Video/Image/Live Inference ---
def run_inference(detector: OpenVINOYOLODetector, source=0, conf_threshold=0.25, flip=False, use_popup=False, video_width=None):
    if isinstance(source, str) and not os.path.exists(source):
        print(f"Downloading sample video: {source}")
        import requests
        url = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/people.mp4"
        r = requests.get(url)
        with open(source, 'wb') as f:
            f.write(r.content)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Failed to open video source: {source}")
        return
    window_name = "YOLOv11x + OpenVINO Detection"
    if use_popup:
        cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
    frame_count = 0
    times = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if flip:
            frame = cv2.flip(frame, 1)
        if video_width:
            scale = video_width / max(frame.shape[:2])
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        start = time.time()
        detections = detector.infer(frame, conf_threshold=conf_threshold)
        frame = detector.draw(frame, detections)
        elapsed = time.time() - start
        times.append(elapsed)
        if len(times) > 200:
            times.pop(0)
        fps = 1.0 / np.mean(times) if times else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        if use_popup:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()

# --- Main Entrypoint ---
if __name__ == "__main__":
    # Choose model: yolo11x or yolo11n, etc.
    MODEL_NAME = "yolo11x"

    DEVICE = "AUTO"  # or "CPU", "GPU"
    # Step 1: Convert model if needed
    ov_xml = convert_yolo_to_openvino(MODEL_NAME)
    # Step 2: Quantize (optional, demo skips actual quantization)
    ov_xml = quantize_openvino_model(ov_xml, MODEL_NAME)
    # Step 3: Create detector
    detector = OpenVINOYOLODetector(ov_xml, device=DEVICE)
    # Step 4: Run on webcam, video, or image
    # Webcam: source=0, Video: source="video.mp4", Image: source="image.jpg"
    run_inference(detector, source=0, conf_threshold=0.25, flip=True, use_popup=True, video_width=1280)
# To run on a video file: run_inference(detector, source="people.mp4", conf_threshold=0.25)
# To run on an image: run_inference(detector, source="image.jpg", conf_threshold=0.25)
# To run async or batch, extend the OpenVINOYOLODetector class with async API as needed.

import os
import time
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional

# Only traffic-related classes for detection
TRAFFIC_CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
    'traffic light', 'stop sign', 'parking meter'
]

class OpenVINOVehicleDetector:
    def __init__(self, model_path: str = None, device: str = "AUTO", use_quantized: bool = False, enable_ocr: bool = False, confidence_threshold: float = 0.4):
        import openvino as ov
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.ocr_reader = None
        self.class_names = TRAFFIC_CLASS_NAMES
        self.performance_stats = {
            'fps': 0,
            'avg_inference_time': 0,
            'frames_processed': 0,
            'backend': f"OpenVINO-{device}",
            'total_detections': 0,
            'detection_rate': 0
        }
        self._inference_times = []
        self._start_time = time.time()
        self._frame_count = 0
        # Model selection logic
        self.model_path = self._find_best_model(model_path, use_quantized)
        self.core = ov.Core()
        self.model = self.core.read_model(self.model_path)
        # Always reshape to static shape before accessing .shape
        self.model.reshape({0: [1, 3, 640, 640]})
        self.input_shape = self.model.inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        self.ov_config = {}
        if device != "CPU":
            # Already reshaped above, so nothing more needed here
            pass
        if "GPU" in device or ("AUTO" in device and "GPU" in self.core.available_devices):
            self.ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
        self.compiled_model = self.core.compile_model(self.model, device, self.ov_config)
        self.output_layer = self.compiled_model.output(0)

    def _find_best_model(self, model_path, use_quantized):
        # Priority: quantized IR > IR > .pt
        search_paths = [
            Path(model_path) if model_path else None,
            Path("yolo11x_openvino_int8_model/yolo11x.xml") if use_quantized else None,
            Path("yolo11x_openvino_model/yolo11x.xml"),
            Path("rcb/yolo11x_openvino_model/yolo11x.xml"),
            Path("yolo11x.xml"),
            Path("rcb/yolo11x.xml"),
            Path("yolo11x.pt"),
            Path("rcb/yolo11x.pt")
        ]
        for p in search_paths:
            if p and p.exists():
                return str(p)
        raise FileNotFoundError("No suitable YOLOv11x model found for OpenVINO.")

    def detect_vehicles(self, frame: np.ndarray, conf_threshold: float = None) -> List[Dict]:
        if conf_threshold is None:
            conf_threshold = self.confidence_threshold
        start = time.time()
        input_tensor = self._preprocess(frame)
        output = self.compiled_model([input_tensor])[self.output_layer]
        detections = self._postprocess(output, frame.shape, conf_threshold)
        elapsed = time.time() - start
        self._inference_times.append(elapsed)
        self._frame_count += 1
        self.performance_stats['frames_processed'] = self._frame_count
        self.performance_stats['total_detections'] += len(detections)
        if len(self._inference_times) > 100:
            self._inference_times.pop(0)
        self.performance_stats['avg_inference_time'] = float(np.mean(self._inference_times)) if self._inference_times else 0
        total_time = time.time() - self._start_time
        self.performance_stats['fps'] = self._frame_count / total_time if total_time > 0 else 0
        return detections

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame, (self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[None]
        return img

    def _postprocess(self, output: np.ndarray, frame_shape, conf_threshold: float) -> List[Dict]:
        # Output: (1, 84, 8400) or (84, 8400) or (8400, 84)
        if output.ndim == 3:
            output = np.squeeze(output)
        if output.shape[0] == 84:
            output = output.T  # (8400, 84)
        boxes = output[:, :4]
        scores = output[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        detections = []
        h, w = frame_shape[:2]
        for i, (box, score, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            if score < conf_threshold:
                continue
            x_c, y_c, bw, bh = box
            # If normalized, scale to input size
            if all(0.0 <= v <= 1.0 for v in box):
                x_c *= self.input_width
                y_c *= self.input_height
                bw *= self.input_width
                bh *= self.input_height
            # Scale to original frame size
            scale_x = w / self.input_width
            scale_y = h / self.input_height
            x_c *= scale_x
            y_c *= scale_y
            bw *= scale_x
            bh *= scale_y
            x1 = int(round(x_c - bw / 2))
            y1 = int(round(y_c - bh / 2))
            x2 = int(round(x_c + bw / 2))
            y2 = int(round(y_c + bh / 2))
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            class_name = TRAFFIC_CLASS_NAMES[class_id] if class_id < len(TRAFFIC_CLASS_NAMES) else str(class_id)
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(score),
                'class_id': int(class_id),
                'class_name': class_name
            })
        return detections

    def draw(self, frame: np.ndarray, detections: List[Dict], box_thickness: int = 2) -> np.ndarray:
        # 80+ visually distinct colors for COCO classes (BGR)
        COCO_COLORS = [
            (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29), (207, 210, 49),
            (72, 249, 10), (146, 204, 23), (61, 219, 134), (26, 147, 52), (0, 212, 187),
            (44, 153, 168), (0, 194, 255), (52, 69, 147), (100, 115, 255), (0, 24, 236),
            (132, 56, 255), (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
            (255, 255, 56), (255, 255, 151), (255, 255, 31), (255, 255, 29), (207, 255, 49),
            (72, 255, 10), (146, 255, 23), (61, 255, 134), (26, 255, 52), (0, 255, 187),
            (44, 255, 168), (0, 255, 255), (52, 255, 147), (100, 255, 255), (0, 255, 236),
            (132, 255, 255), (82, 255, 133), (203, 255, 255), (255, 255, 200), (255, 255, 199),
            (56, 255, 255), (157, 255, 151), (112, 255, 31), (178, 255, 29), (210, 255, 49),
            (249, 255, 10), (204, 255, 23), (219, 255, 134), (147, 255, 52), (212, 255, 187),
            (153, 255, 168), (194, 255, 255), (69, 255, 147), (115, 255, 255), (24, 255, 236),
            (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49),
            (249, 72, 10), (204, 146, 23), (219, 61, 134), (147, 26, 52), (212, 0, 187),
            (153, 44, 168), (194, 0, 255), (69, 52, 147), (115, 100, 255), (24, 0, 236),
            (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49),
            (249, 72, 10), (204, 146, 23), (219, 61, 134), (147, 26, 52), (212, 0, 187),
            (153, 44, 168), (194, 0, 255), (69, 52, 147), (115, 100, 255), (24, 0, 236),
            (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49)
        ]
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class_name']} {det['confidence']:.2f}"
            color = COCO_COLORS[det['class_id'] % len(COCO_COLORS)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
            cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

# --- Video/Image/Live Inference ---
def run_inference(detector: OpenVINOYOLODetector, source=0, conf_threshold=0.25, flip=False, use_popup=False, video_width=None):
    if isinstance(source, str) and not os.path.exists(source):
        print(f"Downloading sample video: {source}")
        import requests
        url = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/people.mp4"
        r = requests.get(url)
        with open(source, 'wb') as f:
            f.write(r.content)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Failed to open video source: {source}")
        return
    window_name = "YOLOv11x + OpenVINO Detection"
    if use_popup:
        cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
    frame_count = 0
    times = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if flip:
            frame = cv2.flip(frame, 1)
        if video_width:
            scale = video_width / max(frame.shape[:2])
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        start = time.time()
        detections = detector.infer(frame, conf_threshold=conf_threshold)
        frame = detector.draw(frame, detections)
        elapsed = time.time() - start
        times.append(elapsed)
        if len(times) > 200:
            times.pop(0)
        fps = 1.0 / np.mean(times) if times else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        if use_popup:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()

# --- Main Entrypoint ---
if __name__ == "__main__":
    # Choose model: yolo11x or yolo11n, etc.
    MODEL_NAME = "yolo11x"

    DEVICE = "AUTO"  # or "CPU", "GPU"
    # Step 1: Convert model if needed
    ov_xml = convert_yolo_to_openvino(MODEL_NAME)
    # Step 2: Quantize (optional, demo skips actual quantization)
    ov_xml = quantize_openvino_model(ov_xml, MODEL_NAME)
    # Step 3: Create detector
    detector = OpenVINOYOLODetector(ov_xml, device=DEVICE)
    # Step 4: Run on webcam, video, or image
    # Webcam: source=0, Video: source="video.mp4", Image: source="image.jpg"
    run_inference(detector, source=0, conf_threshold=0.25, flip=True, use_popup=True, video_width=1280)
# To run on a video file: run_inference(detector, source="people.mp4", conf_threshold=0.25)
# To run on an image: run_inference(detector, source="image.jpg", conf_threshold=0.25)
# To run async or batch, extend the OpenVINOYOLODetector class with async API as needed.

import os
import time
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional

# Only traffic-related classes for detection
TRAFFIC_CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
    'traffic light', 'stop sign', 'parking meter'
]

class OpenVINOVehicleDetector:
    def __init__(self, model_path: str = None, device: str = "AUTO", use_quantized: bool = False, enable_ocr: bool = False, confidence_threshold: float = 0.4):
        import openvino as ov
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.ocr_reader = None
        self.class_names = TRAFFIC_CLASS_NAMES
        self.performance_stats = {
            'fps': 0,
            'avg_inference_time': 0,
            'frames_processed': 0,
            'backend': f"OpenVINO-{device}",
            'total_detections': 0,
            'detection_rate': 0
        }
        self._inference_times = []
        self._start_time = time.time()
        self._frame_count = 0
        # Model selection logic
        self.model_path = self._find_best_model(model_path, use_quantized)
        self.core = ov.Core()
        self.model = self.core.read_model(self.model_path)
        # Always reshape to static shape before accessing .shape
        self.model.reshape({0: [1, 3, 640, 640]})
        self.input_shape = self.model.inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        self.ov_config = {}
        if device != "CPU":
            # Already reshaped above, so nothing more needed here
            pass
        if "GPU" in device or ("AUTO" in device and "GPU" in self.core.available_devices):
            self.ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
        self.compiled_model = self.core.compile_model(self.model, device, self.ov_config)
        self.output_layer = self.compiled_model.output(0)

    def _find_best_model(self, model_path, use_quantized):
        # Priority: quantized IR > IR > .pt
        search_paths = [
            Path(model_path) if model_path else None,
            Path("yolo11x_openvino_int8_model/yolo11x.xml") if use_quantized else None,
            Path("yolo11x_openvino_model/yolo11x.xml"),
            Path("rcb/yolo11x_openvino_model/yolo11x.xml"),
            Path("yolo11x.xml"),
            Path("rcb/yolo11x.xml"),
            Path("yolo11x.pt"),
            Path("rcb/yolo11x.pt")
        ]
        for p in search_paths:
            if p and p.exists():
                return str(p)
        raise FileNotFoundError("No suitable YOLOv11x model found for OpenVINO.")

    def detect_vehicles(self, frame: np.ndarray, conf_threshold: float = None) -> List[Dict]:
        if conf_threshold is None:
            conf_threshold = self.confidence_threshold
        start = time.time()
        input_tensor = self._preprocess(frame)
        output = self.compiled_model([input_tensor])[self.output_layer]
        detections = self._postprocess(output, frame.shape, conf_threshold)
        elapsed = time.time() - start
        self._inference_times.append(elapsed)
        self._frame_count += 1
        self.performance_stats['frames_processed'] = self._frame_count
        self.performance_stats['total_detections'] += len(detections)
        if len(self._inference_times) > 100:
            self._inference_times.pop(0)
        self.performance_stats['avg_inference_time'] = float(np.mean(self._inference_times)) if self._inference_times else 0
        total_time = time.time() - self._start_time
        self.performance_stats['fps'] = self._frame_count / total_time if total_time > 0 else 0
        return detections

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame, (self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[None]
        return img

    def _postprocess(self, output: np.ndarray, frame_shape, conf_threshold: float) -> List[Dict]:
        # Output: (1, 84, 8400) or (84, 8400) or (8400, 84)
        if output.ndim == 3:
            output = np.squeeze(output)
        if output.shape[0] == 84:
            output = output.T  # (8400, 84)
        boxes = output[:, :4]
        scores = output[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        detections = []
        h, w = frame_shape[:2]
        for i, (box, score, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            if score < conf_threshold:
                continue
            x_c, y_c, bw, bh = box
            # If normalized, scale to input size
            if all(0.0 <= v <= 1.0 for v in box):
                x_c *= self.input_width
                y_c *= self.input_height
                bw *= self.input_width
                bh *= self.input_height
            # Scale to original frame size
            scale_x = w / self.input_width
            scale_y = h / self.input_height
            x_c *= scale_x
            y_c *= scale_y
            bw *= scale_x
            bh *= scale_y
            x1 = int(round(x_c - bw / 2))
            y1 = int(round(y_c - bh / 2))
            x2 = int(round(x_c + bw / 2))
            y2 = int(round(y_c + bh / 2))
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            class_name = TRAFFIC_CLASS_NAMES[class_id] if class_id < len(TRAFFIC_CLASS_NAMES) else str(class_id)
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(score),
                'class_id': int(class_id),
                'class_name': class_name
            })
        return detections

    def draw(self, frame: np.ndarray, detections: List[Dict], box_thickness: int = 2) -> np.ndarray:
        # 80+ visually distinct colors for COCO classes (BGR)
        COCO_COLORS = [
            (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29), (207, 210, 49),
            (72, 249, 10), (146, 204, 23), (61, 219, 134), (26, 147, 52), (0, 212, 187),
            (44, 153, 168), (0, 194, 255), (52, 69, 147), (100, 115, 255), (0, 24, 236),
            (132, 56, 255), (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
            (255, 255, 56), (255, 255, 151), (255, 255, 31), (255, 255, 29), (207, 255, 49),
            (72, 255, 10), (146, 255, 23), (61, 255, 134), (26, 255, 52), (0, 255, 187),
            (44, 255, 168), (0, 255, 255), (52, 255, 147), (100, 255, 255), (0, 255, 236),
            (132, 255, 255), (82, 255, 133), (203, 255, 255), (255, 255, 200), (255, 255, 199),
            (56, 255, 255), (157, 255, 151), (112, 255, 31), (178, 255, 29), (210, 255, 49),
            (249, 255, 10), (204, 255, 23), (219, 255, 134), (147, 255, 52), (212, 255, 187),
            (153, 255, 168), (194, 255, 255), (69, 255, 147), (115, 255, 255), (24, 255, 236),
            (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49),
            (249, 72, 10), (204, 146, 23), (219, 61, 134), (147, 26, 52), (212, 0, 187),
            (153, 44, 168), (194, 0, 255), (69, 52, 147), (115, 100, 255), (24, 0, 236),
            (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49),
            (249, 72, 10), (204, 146, 23), (219, 61, 134), (147, 26, 52), (212, 0, 187),
            (153, 44, 168), (194, 0, 255), (69, 52, 147), (115, 100, 255), (24, 0, 236),
            (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49)
        ]
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class_name']} {det['confidence']:.2f}"
            color = COCO_COLORS[det['class_id'] % len(COCO_COLORS)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
            cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

# --- Video/Image/Live Inference ---
def run_inference(detector: OpenVINOYOLODetector, source=0, conf_threshold=0.25, flip=False, use_popup=False, video_width=None):
    if isinstance(source, str) and not os.path.exists(source):
        print(f"Downloading sample video: {source}")
        import requests
        url = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/people.mp4"
        r = requests.get(url)
        with open(source, 'wb') as f:
            f.write(r.content)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Failed to open video source: {source}")
        return
    window_name = "YOLOv11x + OpenVINO Detection"
    if use_popup:
        cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
    frame_count = 0
    times = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if flip:
            frame = cv2.flip(frame, 1)
        if video_width:
            scale = video_width / max(frame.shape[:2])
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        start = time.time()
        detections = detector.infer(frame, conf_threshold=conf_threshold)
        frame = detector.draw(frame, detections)
        elapsed = time.time() - start
        times.append(elapsed)
        if len(times) > 200:
            times.pop(0)
        fps = 1.0 / np.mean(times) if times else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        if use_popup:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()

# --- Main Entrypoint ---
if __name__ == "__main__":
    # Choose model: yolo11x or yolo11n, etc.
    MODEL_NAME = "yolo11x"

    DEVICE = "AUTO"  # or "CPU", "GPU"
    # Step 1: Convert model if needed
    ov_xml = convert_yolo_to_openvino(MODEL_NAME)
    # Step 2: Quantize (optional, demo skips actual quantization)
    ov_xml = quantize_openvino_model(ov_xml, MODEL_NAME)
    # Step 3: Create detector
    detector = OpenVINOYOLODetector(ov_xml, device=DEVICE)
    # Step 4: Run on webcam, video, or image
    # Webcam: source=0, Video: source="video.mp4", Image: source="image.jpg"
    run_inference(detector, source=0, conf_threshold=0.25, flip=True, use_popup=True, video_width=1280)
# To run on a video file: run_inference(detector, source="people.mp4", conf_threshold=0.25)
# To run on an image: run_inference(detector, source="image.jpg", conf_threshold=0.25)
# To run async or batch, extend the OpenVINOYOLODetector class with async API as needed.

import os
import time
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional

# Only traffic-related classes for detection
TRAFFIC_CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
    'traffic light', 'stop sign', 'parking meter'
]

class OpenVINOVehicleDetector:
    def __init__(self, model_path: str = None, device: str = "AUTO", use_quantized: bool = False, enable_ocr: bool = False, confidence_threshold: float = 0.4):
        import openvino as ov
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.ocr_reader = None
        self.class_names = TRAFFIC_CLASS_NAMES
        self.performance_stats = {
            'fps': 0,
            'avg_inference_time': 0,
            'frames_processed': 0,
            'backend': f"OpenVINO-{device}",
            'total_detections': 0,
            'detection_rate': 0
        }
        self._inference_times = []
        self._start_time = time.time()
        self._frame_count = 0
        # Model selection logic
        self.model_path = self._find_best_model(model_path, use_quantized)
        self.core = ov.Core()
        self.model = self.core.read_model(self.model_path)
        # Always reshape to static shape before accessing .shape
        self.model.reshape({0: [1, 3, 640, 640]})
        self.input_shape = self.model.inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        self.ov_config = {}
        if device != "CPU":
            # Already reshaped above, so nothing more needed here
            pass
        if "GPU" in device or ("AUTO" in device and "GPU" in self.core.available_devices):
            self.ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
        self.compiled_model = self.core.compile_model(self.model, device, self.ov_config)
        self.output_layer = self.compiled_model.output(0)

    def _find_best_model(self, model_path, use_quantized):
        # Priority: quantized IR > IR > .pt
        search_paths = [
            Path(model_path) if model_path else None,
            Path("yolo11x_openvino_int8_model/yolo11x.xml") if use_quantized else None,
            Path("yolo11x_openvino_model/yolo11x.xml"),
            Path("rcb/yolo11x_openvino_model/yolo11x.xml"),
            Path("yolo11x.xml"),
            Path("rcb/yolo11x.xml"),
            Path("yolo11x.pt"),
            Path("rcb/yolo11x.pt")
        ]
        for p in search_paths:
            if p and p.exists():
                return str(p)
        raise FileNotFoundError("No suitable YOLOv11x model found for OpenVINO.")

    def detect_vehicles(self, frame: np.ndarray, conf_threshold: float = None) -> List[Dict]:
        if conf_threshold is None:
            conf_threshold = self.confidence_threshold
        start = time.time()
        input_tensor = self._preprocess(frame)
        output = self.compiled_model([input_tensor])[self.output_layer]
        detections = self._postprocess(output, frame.shape, conf_threshold)
        elapsed = time.time() - start
        self._inference_times.append(elapsed)
        self._frame_count += 1
        self.performance_stats['frames_processed'] = self._frame_count
        self.performance_stats['total_detections'] += len(detections)
        if len(self._inference_times) > 100:
            self._inference_times.pop(0)
        self.performance_stats['avg_inference_time'] = float(np.mean(self._inference_times)) if self._inference_times else 0
        total_time = time.time() - self._start_time
        self.performance_stats['fps'] = self._frame_count / total_time if total_time > 0 else 0
        return detections

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame, (self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[None]
        return img

    def _postprocess(self, output: np.ndarray, frame_shape, conf_threshold: float) -> List[Dict]:
        # Output: (1, 84, 8400) or (84, 8400) or (8400, 84)
        if output.ndim == 3:
            output = np.squeeze(output)
        if output.shape[0] == 84:
            output = output.T  # (8400, 84)
        boxes = output[:, :4]
        scores = output[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        detections = []
        h, w = frame_shape[:2]
        for i, (box, score, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            if score < conf_threshold:
                continue
            x_c, y_c, bw, bh = box
            # If normalized, scale to input size
            if all(0.0 <= v <= 1.0 for v in box):
                x_c *= self.input_width
                y_c *= self.input_height
                bw *= self.input_width
                bh *= self.input_height
            # Scale to original frame size
            scale_x = w / self.input_width
            scale_y = h / self.input_height
            x_c *= scale_x
            y_c *= scale_y
            bw *= scale_x
            bh *= scale_y
            x1 = int(round(x_c - bw / 2))
            y1 = int(round(y_c - bh / 2))
            x2 = int(round(x_c + bw / 2))
            y2 = int(round(y_c + bh / 2))
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            class_name = TRAFFIC_CLASS_NAMES[class_id] if class_id < len(TRAFFIC_CLASS_NAMES) else str(class_id)
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(score),
                'class_id': int(class_id),
                'class_name': class_name
            })
        return detections

    def draw(self, frame: np.ndarray, detections: List[Dict], box_thickness: int = 2) -> np.ndarray:
        # 80+ visually distinct colors for COCO classes (BGR)
        COCO_COLORS = [
            (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29), (207, 210, 49),
            (72, 249, 10), (146, 204, 23), (61, 219, 134), (26, 147, 52), (0, 212, 187),
            (44, 153, 168), (0, 194, 255), (52, 69, 147), (100, 115, 255), (0, 24, 236),
            (132, 56, 255), (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
            (255, 255, 56), (255, 255, 151), (255, 255, 31), (255, 255, 29), (207, 255, 49),
            (72, 255, 10), (146, 255, 23), (61, 255, 134), (26, 255, 52), (0, 255, 187),
            (44, 255, 168), (0, 255, 255), (52, 255, 147), (100, 255, 255), (0, 255, 236),
            (132, 255, 255), (82, 255, 133), (203, 255, 255), (255, 255, 200), (255, 255, 199),
            (56, 255, 255), (157, 255, 151), (112, 255, 31), (178, 255, 29), (210, 255, 49),
            (249, 255, 10), (204, 255, 23), (219, 255, 134), (147, 255, 52), (212, 255, 187),
            (153, 255, 168), (194, 255, 255), (69, 255, 147), (115, 255, 255), (24, 255, 236),
            (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49),
            (249, 72, 10), (204, 146, 23), (219, 61, 134), (147, 26, 52), (212, 0, 187),
            (153, 44, 168), (194, 0, 255), (69, 52, 147), (115, 100, 255), (24, 0, 236),
            (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49),
            (249, 72, 10), (204, 146, 23), (219, 61, 134), (147, 26, 52), (212, 0, 187),
            (153, 44, 168), (194, 0, 255), (69, 52, 147), (115, 100, 255), (24, 0, 236),
            (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49)
        ]
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class_name']} {det['confidence']:.2f}"
            color = COCO_COLORS[det['class_id'] % len(COCO_COLORS)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
            cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

# --- Video/Image/Live Inference ---
def run_inference(detector: OpenVINOYOLODetector, source=0, conf_threshold=0.25, flip=False, use_popup=False, video_width=None):
    if isinstance(source, str) and not os.path.exists(source):
        print(f"Downloading sample video: {source}")
        import requests
        url = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/people.mp4"
        r = requests.get(url)
        with open(source, 'wb') as f:
            f.write(r.content)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Failed to open video source: {source}")
        return
    window_name = "YOLOv11x + OpenVINO Detection"
    if use_popup:
        cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
    frame_count = 0
    times = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if flip:
            frame = cv2.flip(frame, 1)
        if video_width:
            scale = video_width / max(frame.shape[:2])
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        start = time.time()
        detections = detector.infer(frame, conf_threshold=conf_threshold)
        frame = detector.draw(frame, detections)
        elapsed = time.time() - start
        times.append(elapsed)
        if len(times) > 200:
            times.pop(0)
        fps = 1.0 / np.mean(times) if times else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        if use_popup:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()

# --- Main Entrypoint ---
if __name__ == "__main__":
    # Choose model: yolo11x or yolo11n, etc.
    MODEL_NAME = "yolo11x"

    DEVICE = "AUTO"  # or "CPU", "GPU"
    # Step 1: Convert model if needed
    ov_xml = convert_yolo_to_openvino(MODEL_NAME)
    # Step 2: Quantize (optional, demo skips actual quantization)
    ov_xml = quantize_openvino_model