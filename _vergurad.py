from common.camera import Camera
import cv2

camera= Camera(0, 640, 480)

while True:
    frame = camera.get_frame()
    cv2.imshow('T',frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()