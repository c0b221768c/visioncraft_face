from src.camera import Camera
import cv2
camera = Camera(3)

import cv2

while True:
    frame = camera.get_frame()
    cv2.imshow("Camera00", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESCキーで終了
        break