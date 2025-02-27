import time
import cv2
import os

from api.sender import SenderTCP
from camera import Camera
from config import config
from detection import FaceDetector

sender = SenderTCP()
camera = Camera(0)
detector = FaceDetector()

face_persist_time = None

print("🚀 Starting identification...")
while True:
    frame = camera.get_frame()
    if frame is None:
        continue

    face = detector.detect_face(frame)
    current_time = time.time()
    if face is not None:
        x1, x2, y1, y2 = face
        face_size = (x2 - x1) * (y2 - y1)
        if face_size > config.FACE_SIZE_THRESHOLD:
            if face_persist_time is None:
                face_persist_time = current_time
            else:
                elapsed_time = current_time - face_persist_time
                if elapsed_time > config.FACE_PERSIST_THRESHOLD:
                    sender.send_request("attract", "hello", 0)
                    print("📡 Data sent: 'hello'")
                    face_persist_time = None
        else:
            face_persist_time = None
    else:
        face_persist_time = None

    cv2.imshow("Camera00", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

camera.release()
cv2.destroyAllWindows()
print("🛑 Stopped identification.")