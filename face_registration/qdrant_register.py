import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import uuid

import cv2
import numpy as np

from common.camera import Camera
from common.detection import FaceDetector
from common.recognition import FaceRecognition
from common.utils import init_db, save_feature


def register_face(
    user_uuid, camera: Camera, detector: FaceDetector, recognizer: FaceRecognition
):
    print(f"UUID: {user_uuid}")

    collected_features = []
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue

        faces = detector.detect_face(frame)
        if not faces:
            cv2.imshow("Face Registration", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # 顔の領域を切り抜く
        x1, y1, x2, y2 = faces[0]
        face_crop = frame[y1:y2, x1:x2]
        feature = recognizer.extract_feature(face_crop)
        collected_features.append(feature)

        print(f"Registering... Collected {len(collected_features)} features")
        if len(collected_features) >= 5:
            break

    camera.release()
    cv2.destroyAllWindows()

    if collected_features:
        avg_feature = np.mean(collected_features, axis=0)  # 特徴量の平均をとる
        save_feature(user_uuid, avg_feature)  # Qdrant に保存
        print(f"Success: UUID={user_uuid}")


if __name__ == "__main__":
    init_db()  # Qdrant のコレクションを初期化
    camera = Camera(0, 640, 480)
    detector = FaceDetector()
    recognizer = FaceRecognition("models/face_recognition.onnx")

    user_uuid = str(uuid.uuid4())  # 新規 UUID を生成
    register_face(user_uuid, camera, detector, recognizer)
