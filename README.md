# OpenVINO Traffic Monitoring System

A modular, production-ready, and advanced traffic monitoring application using OpenVINO, YOLOv11x, and Streamlit. This system provides real-time vehicle detection, traffic violation analytics, and a modern web UI for deployment and analytics.

---

## Features

- **Full OpenVINO Pipeline**: Efficient IR model loading, CPU inference, async support, and best practices.
- **YOLOv11x/YOLOv5 Support**: Fast, accurate object detection for traffic scenes.
- **Real-Time Analytics**: Live FPS, rolling averages, per-frame inference time, detection/violation counts.
- **Advanced Streamlit UI**: Sidebar controls, live video, stats, rolling FPS chart, top banner, modular layout.
- **Flexible Input**: Webcam, video file upload, or image upload.
- **Modular Codebase**: Clean separation of detection, violation logic, and utilities.
- **Robust Error Handling**: Graceful handling of model, video, and inference errors.
- **Export & Logging**: Downloadable CSVs for detections, violations, and performance stats.

---

## File Overview

### `app1.py` / `app2.py`

- **Main Streamlit app**
- Loads OpenVINO model, handles video/image input, runs detection, displays annotated frames and analytics.
- Modular functions for model loading, preprocessing, postprocessing, annotation, and UI.
- Real-time stats: FPS, inference time, detection/violation count, rolling charts.
- Advanced UI: sidebar controls, top banner, live video, stats, export buttons.

### `detection_openvino.py`

- **Detection logic using OpenVINO**
- Loads IR model, handles preprocessing (resize, normalize, transpose), runs inference, and postprocesses outputs (NMS, confidence filtering).
- Provides a `TrafficDetector` class with methods like `detect_vehicles(frame, conf_threshold)` and `get_performance_stats()`.
- Designed for easy integration with Streamlit or other apps.

### `violation_openvino.py`

- **Traffic violation detection logic**
- Implements rules for red light, stop sign, speed, and other violations.
- Accepts detection results and frame/time context, returns violation dictionaries (type, description, severity, confidence, etc.).
- Designed to be modular and extensible for new violation types.

### `utils.py`

- **Utility functions**
- Frame resizing, annotation helpers, CSV export, configuration management, and other shared helpers.
- Used by both detection and violation modules for DRY code.

---

## Quick Start

### 1. Install Requirements

```powershell
pip install -r requirements-openvino.txt
```

### 2. Run the App

```powershell
streamlit run app2.py
```

### 3. Usage

- Use the sidebar to select input (webcam, video, image), adjust thresholds, and start/stop detection.
- View live video, stats, and analytics in the main panel.
- Download detection/violation logs as CSV for further analysis.

---

## Advanced Usage

- **Model Hot-Swap**: Place your OpenVINO IR model in the root or `openvino_models/` folder. The app will auto-detect.
- **Async Inference**: The detection module is designed for async inference (see `detection_openvino.py`).
- **Custom Violations**: Extend `violation_openvino.py` to add new traffic rules.
- **Performance Tuning**: Adjust batch size, device, and thresholds in the sidebar or config.

---

## Best Practices & Recommendations

- Use the latest OpenVINO and a supported CPU for best performance.
- For production, run behind a secure web server and use HTTPS.
- Monitor resource usage if running on edge devices.
- Regularly update your detection and violation logic for new scenarios.

---

## Example Directory Structure

```
├── app1.py / app2.py
├── detection_openvino.py
├── violation_openvino.py
├── utils.py
├── requirements-openvino.txt
├── openvino_models/
│   └── yolo11n.xml
│   └── yolo11n.bin
├── rcb/
│   └── yolo11x_openvino_model/
│       └── yolo11x.xml
│       └── yolo11x.bin
├── ...
```

---

## Credits

- [OpenVINO Toolkit](https://github.com/openvinotoolkit/openvino)
- [YOLOv5/YOLOv11x](https://github.com/ultralytics/yolov5)
- [Streamlit](https://streamlit.io/)

---

## License

This project is for research and educational purposes. See individual files for license details.
