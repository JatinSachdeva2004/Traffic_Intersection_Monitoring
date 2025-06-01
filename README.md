#Traffic Intersection Monitoring System with OpenVINO

This project implements a real-time traffic monitoring solution that detects vehicles, pedestrians, and traffic violations at intersections using object detection models optimized with OpenVINO. It features a PyQt5-based dashboard for visualization and control, integrates synthetic data generation using CARLA, and supports enhanced scene understanding through vision-language models.

## Problem Statement

The system monitors traffic intersections to identify and track vehicles, pedestrians, and cyclists in real-time. It collects traffic statistics and detects violations such as red-light running and jaywalking. The focus is on efficient deployment at the edge using Intel hardware.

## Objectives

- Detect vehicles, pedestrians, and cyclists using object detection
- Monitor and record traffic violations in real-time
- Display detection results and statistics through a graphical interface
- Enable model deployment using OpenVINO for optimized inference
- Generate and annotate synthetic traffic data using CARLA
- Integrate visual reasoning capabilities through vision-language models

## Training and Optimization

1. **Model Training**  
   The YOLOv12 model is trained using PyTorch with labeled image data representing traffic scenes.

2. **Export Pipeline**  
   The trained model is exported to ONNX format, and then converted to OpenVINO's Intermediate Representation (IR) format.

3. **Optimization**  
   Post-training quantization is applied to convert the model from FP32 to INT8, improving inference speed while maintaining accuracy.

4. **Deployment**  
   OpenVINO's InferRequest API is used for asynchronous inference, enabling efficient frame-by-frame processing suitable for real-time applications.

## Synthetic Data Generation

CARLA is used to simulate traffic intersections with accurate layouts, signage, and weather variations. It supports:

- Scene diversity through environmental changes (rain, fog, glare, nighttime)
- Simulation of pedestrian and vehicle behaviors (red-light running, jaywalking)
- Automatic annotation of bounding boxes and class labels for use with object detection models

## Vision-Language Integration

Two models are integrated to enhance scene understanding:

- **BLIP-2**: Automatically generates text summaries of visual scenes (e.g., “A vehicle is crossing the red light”)
- **LLaVA**: Enables question-answering over video frames (e.g., “Why was the pedestrian flagged?”)

These tools allow human operators to interact with the system more effectively by supporting natural language explanations and queries.

## PyQt5-Based Dashboard

The dashboard enables real-time interaction with the monitoring system and includes:

- Live video feed with overlayed bounding boxes
- Detection tags for pedestrians, vehicles, and violators
- Violation statistics and traffic flow metrics
- Controls for switching between camera sources and simulated environments
- High-performance rendering using QPainter for dynamic visual updates
