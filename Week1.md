# GSOC-25: Traffic Intersection Monitoring with OpenVINO

This project develops a real-time system to detect traffic objects at intersections. It uses YOLOv11 and YOLOv12 deep learning models optimized with OpenVINO to identify vehicles, pedestrians, and traffic signs efficiently on Intel hardware.



## Current Progress (Week 1)

- Built the main detection pipeline  
- Tested different YOLO models for accuracy and speed  
- Created vehicle classification based on size and shape  
- Developed image processing and visualization tools  
- Added tracking to maintain object consistency between frames  
- Implemented filtering to remove false positives and overlapping detections  


## Features

- Train custom YOLOv12n models using traffic data from the COCO dataset  
- Convert models from PyTorch format to OpenVINO IR format  
- Quantize models to INT8 for faster inference without losing accuracy  
- Run detection on images, video files, and webcam streams  
- Detect common traffic classes such as cars, trucks, pedestrians, and traffic lights  
- Deploy models on CPU, GPU, and other OpenVINO-supported devices  
