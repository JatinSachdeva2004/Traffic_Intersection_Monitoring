# Utility for drawing detections, tracks, and violations on frames
import utils

def enhanced_annotate_frame(app, frame, detections, violations):
    import cv2
    import numpy as np
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return np.zeros((300, 300, 3), dtype=np.uint8)
    annotated_frame = frame.copy()
    if detections is None:
        detections = []
    if violations is None:
        violations = []
    if len(detections) > 0:
        if hasattr(app, 'tracker') and app.tracker:
            try:
                ds_dets = []
                for det in detections:
                    if 'bbox' not in det:
                        continue
                    try:
                        bbox = det['bbox']
                        if len(bbox) < 4:
                            continue
                        x1, y1, x2, y2 = bbox
                        w = x2 - x1
                        h = y2 - y1
                        if w <= 0 or h <= 0:
                            continue
                        conf = det.get('confidence', 0.0)
                        class_name = det.get('class_name', 'unknown')
                        ds_dets.append(([x1, y1, w, h], conf, class_name))
                    except Exception:
                        continue
                if ds_dets:
                    tracks = app.tracker.update_tracks(ds_dets, frame=frame.copy())
                    for track in tracks:
                        if not track.is_confirmed():
                            continue
                        tid = track.track_id
                        ltrb = track.to_ltrb()
                        for det in detections:
                            if 'bbox' not in det:
                                continue
                            try:
                                bbox = det['bbox']
                                if len(bbox) < 4:
                                    continue
                                dx1, dy1, dx2, dy2 = bbox
                                iou = utils.bbox_iou((dx1, dy1, dx2, dy2), tuple(map(int, ltrb)))
                                if iou > 0.5:
                                    det['track_id'] = tid
                                    break
                            except Exception:
                                continue
            except Exception:
                pass
    try:
        show_labels = app.config.get('display', {}).get('show_labels', True)
        show_confidence = app.config.get('display', {}).get('show_confidence', True)
        annotated_frame = utils.draw_detections(annotated_frame, detections, show_labels, show_confidence)
        annotated_frame = utils.draw_violations(annotated_frame, violations)
        return annotated_frame
    except Exception:
        return frame.copy()
