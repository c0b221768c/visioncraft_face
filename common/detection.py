import cv2
import numpy as np
from retinaface import RetinaFace

class FaceDetector:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    
    def detect_face(self, frame):
        results = RetinaFace.detect_faces(frame)
        if not results:
            return []
        boxes = []

        for key, face_info in results.items():
            confidence = face_info["score"]
            if confidence >= self.threshold:
                x1, y1, x2, y2 = face_info["facial_area"]
                boxes.append([x1, y1, x2, y2])
        
        if boxes:
            boxes = sorted(boxes, key=lambda b:(b[2] - b[0]) * (b[3] - b[1]), reverse=True)
            return [boxes[0]]
        return []