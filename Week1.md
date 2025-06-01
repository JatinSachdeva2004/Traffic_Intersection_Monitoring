GSOC-25-Traffic_Intersection_Monitoring with OpenVino

# GSOC-25: Traffic Intersection Monitoring with OpenVINO

This system leverages **YOLOv11** and **YOLOv12** deep learning models for accurate detection and classification of traffic-related objects.

Week 1 Summary
Established the core detection pipeline and evaluated performance across model variants.

Key Work Completed
1. **Integrated YOLOv8 and YOLOv11 models** for traffic object detection.
2. **Created class mapping from COCO** → traffic-specific labels (e.g., car, truck, pedestrian, traffic light).
3. **Implemented size-, shape-, and distance-aware vehicle classification**.
4. **Built an image processing pipeline** for enhanced detection and visualization.
5. **Developed a temporal tracking module** to improve detection stability across video frames.
6. **Added advanced post-processing** for:
   - Off-road object filtering
   - Overlapping bounding box suppression




