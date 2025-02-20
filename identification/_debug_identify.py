# debug_identify.py
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import queue
import threading

from _gui import display_gui
from config import NUM_CAMERAS
from identify_core import identify

from api.sender import SenderTCP
from common.camera import Camera
from common.detection import FaceDetector
from common.recognition import FaceRecognition

# データ送信の有効・無効を切り替える（デフォルトはFalse）
SEND_DATA = False

frame_queues = {i: queue.Queue(maxsize=1) for i in range(NUM_CAMERAS)}
stop_event = threading.Event()


if __name__ == "__main__":
    cameras = [Camera(i, 640, 480) for i in range(NUM_CAMERAS)]
    sender = SenderTCP("172.16.103.17", 8080) if SEND_DATA else None
    detector = FaceDetector()
    recognizer = FaceRecognition("models/face_recognition.onnx")

    threads = [
        threading.Thread(
            target=identify,
            args=(sender, cameras[i], detector, recognizer, i, frame_queues, SEND_DATA),
            daemon=True,
        )
        for i in range(NUM_CAMERAS)
    ]

    for thread in threads:
        thread.start()

    gui_thread = threading.Thread(
        target=display_gui, args=(frame_queues, stop_event), daemon=True
    )
    gui_thread.start()

    try:
        while not stop_event.is_set():
            pass  # メインスレッドを維持
    except KeyboardInterrupt:
        stop_event.set()
