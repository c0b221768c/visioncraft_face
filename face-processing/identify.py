import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from collections import deque, Counter
from common.camera import Camera
from common.detection import FaceDetector
from common.recognition import FaceRecognition
from common.utils import load_all_features, cosine_similarity

THRESHOLD = 0.6
BUFFER_SIZE = 10

def identify(camera:Camera, detector:FaceDetector, recognizer:FaceRecognition):
    print("O : Start")

    id_buffer = deque(maxlen=BUFFER_SIZE)
    uuids, feature_db = load_all_features()

    while True:
        frame = camera.get_frame()
        if frame is None:
            continue

        faces = detector.detect_face(frame)
        if not faces:
            print("X : Can not detect")
            continue

        x1, y1, x2, y2 = faces[0]
        face_crop = frame[y1:y2, x1:x2]
        feature = recognizer.extract_feature(face_crop)

        best_match_uuid = "Unknown"
        best_match_sim = 0.0

        for db_uuid, db_feature in zip(uuids, feature_db):
            sim = cosine_similarity(feature, db_feature)
            if sim > best_match_sim:
                best_match_sim = sim
                best_match_uuid = db_uuid
        
        if best_match_sim < THRESHOLD:
            best_match_uuid = "Unknown"
        
        id_buffer.append(best_match_uuid)

        if len(id_buffer) == BUFFER_SIZE:
            most_common_id, count = Counter(id_buffer).most_common(1)[0]
            if count >= BUFFER_SIZE*0.7:
                final_id = most_common_id
            else:
                final_id = "uncertain"
        
            print(f"Success: {final_id} ({count}/{BUFFER_SIZE})")

if __name__ == "__main__":
    camera = Camera(0, 640, 480)
    detector = FaceDetector()
    recognizer = FaceRecognition("models/face_recognition.onnx")

    identify(camera, detector, recognizer)
