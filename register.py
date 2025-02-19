import cv2
import uuid
import numpy as np
from common.camera import Camera
from common.detection import FaceDetector
from common.recognition import FaceRecognition
from common.utils import init_qdrant, save_feature

def register_face(user_uuid, camera: Camera, detector: FaceDetector, recognizer: FaceRecognition):
    print(f"UUID: {user_uuid}")
    collected_features = []

    while True:
        frame = camera.get_frame()
        if frame is None:
            continue

        faces = detector.detect_faces(frame)
        if not faces:
            cv2.imshow("Face Registration", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        x1, y1, x2, y2 = faces[0]
        face_crop = frame[y1:y2, x1:x2]
        feature = recognizer.extract_feature(face_crop)

        if feature is not None:
            collected_features.append(feature)
            print(f"Captured Feature {len(collected_features)}/5")

        # 顔ボックスを描画
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Face Registration", frame)

        if len(collected_features) >= 5:
            break

    camera.release()
    cv2.destroyAllWindows()

    if collected_features:
        avg_feature = np.mean(collected_features, axis=0)
        save_feature(user_uuid, avg_feature)
        print(f"Success: UUID={user_uuid}")

if __name__ == "__main__":
    init_qdrant()
    camera = Camera(0, 640, 480)
    detector = FaceDetector()
    recognizer = FaceRecognition()

    user_uuid = str(uuid.uuid4())
    register_face(user_uuid, camera, detector, recognizer)
