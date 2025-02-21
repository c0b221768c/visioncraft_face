# identify.py

import signal
import threading

from identify_core import identify

from api.sender import SenderTCP
from common.camera import Camera
from common.config import config
from common.detection import FaceDetector
from common.recognition import FaceRecognition

stop_event = threading.Event()


def signal_handler(sig, frame):
    print("\n🔴 Ctrl + C detected! Stopping all threads...")
    stop_event.set()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    cameras = [Camera(i, 640, 480) for i in range(config.NUM_CAMERAS)]
    sender = SenderTCP("172.16.103.17", 8080)
    detector = FaceDetector()
    recognizer = FaceRecognition("models/face_recognition.onnx")

    threads = [
        threading.Thread(
            target=identify,
            args=(sender, cameras[i], detector, recognizer, i),
            daemon=True,
        )
        for i in range(config.NUM_CAMERAS)
    ]

    for thread in threads:
        thread.start()

    try:
        while not stop_event.is_set():
            pass  # メインスレッドが終了しないようにする
    except KeyboardInterrupt:
        stop_event.set()
