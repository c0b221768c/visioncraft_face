import os
import queue
import signal
import sys
import threading
import time
from collections import Counter, deque

import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.sender import SenderTCP
from common.camera import Camera
from common.detection import FaceDetector
from common.recognition import FaceRecognition
from common.utils import is_uuid, search_feature

THRESHOLD = 0.6
BUFFER_SIZE = 10
FRAME_SKIP = 5  # 5フレームごとに処理を実行
TIMEOUT = 5  # 5秒間認識がなかったらリセット

# GUI表示用のフレームを保存するキュー（各カメラ用）
frame_queues = {i: queue.Queue(maxsize=1) for i in range(4)}

# グローバル変数（スレッド間で共有）
active_camera_id = None  # 現在アクティブなカメラ
active_uuid = None  # 現在認識されているユーザー
last_detected_time = None  # 最後に認識された時間
lock = threading.Lock()  # スレッド間の排他制御用ロック

# 停止イベント（`Ctrl + C` を検出してすべてのスレッドを終了する）
stop_event = threading.Event()


def identify(
    sender: SenderTCP,
    camera: Camera,
    detector: FaceDetector,
    recognizer: FaceRecognition,
    machine_id: int,
):
    """
    各カメラ（machine_idごと）に識別処理を実行
    """
    global active_camera_id, active_uuid, last_detected_time

    print(f"🎥 Camera {machine_id} Started")

    id_buffer = deque(maxlen=BUFFER_SIZE)
    frame_count = 0  # フレームカウント

    while not stop_event.is_set():
        frame = camera.get_frame()
        if frame is None:
            continue

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue  # 設定したフレーム間隔でのみ処理

        faces = detector.detect_face(frame)
        if not faces:
            frame_queues[machine_id].queue.clear()
            frame_queues[machine_id].put(frame)
            continue

        x1, y1, x2, y2 = faces[0]

        if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
            continue

        face_crop = frame[y1:y2, x1:x2]

        if face_crop is None or face_crop.size == 0:
            continue

        feature = recognizer.extract_feature(face_crop)

        # Qdrant に問い合わせて検索
        best_match_uuid, best_match_sim = search_feature(feature)

        if best_match_sim < THRESHOLD:
            best_match_uuid = "Unknown"

        id_buffer.append(best_match_uuid)

        with lock:
            current_time = time.time()

            if len(id_buffer) == BUFFER_SIZE:
                most_common_id, count = Counter(id_buffer).most_common(1)[0]
                if count >= BUFFER_SIZE * 0.7:
                    detected_uuid = most_common_id
                else:
                    detected_uuid = "uncertain"

                # **送信条件の判定**
                if detected_uuid is not None and is_uuid(detected_uuid):
                    if active_uuid is None:
                        # **初回認識**
                        active_uuid = detected_uuid
                        active_camera_id = machine_id
                        last_detected_time = current_time
                        sender.send_request(machine_id=machine_id, uuid=detected_uuid)
                        print(f"✅ Sent from Camera {machine_id}: {detected_uuid}")

                    elif (
                        active_uuid == detected_uuid and active_camera_id != machine_id
                    ):
                        # **既存ユーザーがカメラを移動**
                        active_camera_id = machine_id
                        last_detected_time = current_time
                        sender.send_request(machine_id=machine_id, uuid=detected_uuid)
                        print(f"✅ Sent from Camera {machine_id}: {detected_uuid}")

                # **一定時間経過でリセット**
                if (
                    active_uuid is not None
                    and current_time - last_detected_time > TIMEOUT
                ):
                    print("🕒 Timeout: Reset active user")
                    active_uuid = None
                    active_camera_id = None

        # フレームをGUI用のキューに送信
        label = f"ID: {best_match_uuid} (Sim: {best_match_sim:.2f})"
        cv2.putText(
            frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        frame_queues[machine_id].queue.clear()
        frame_queues[machine_id].put(frame)

        time.sleep(0.1)


def display_gui():
    try:
        while not stop_event.is_set():
            for machine_id in range(4):
                if not frame_queues[machine_id].empty():
                    frame = frame_queues[machine_id].get()
                    if frame is not None:
                        cv2.imshow(f"Camera {machine_id}", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()
                break
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        cv2.destroyAllWindows()


def signal_handler(sig, frame):
    """`Ctrl + C` が押されたら全スレッドを停止"""
    print("\n🔴 Ctrl + C detected! Stopping all threads...")
    stop_event.set()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    cameras = [
        Camera(0, 640, 480),
        Camera(1, 640, 480),
        Camera(2, 640, 480),
        Camera(3, 640, 480),
    ]

    sender = SenderTCP("172.16.103.17", 8080)
    detector = FaceDetector()
    recognizer = FaceRecognition("models/face_recognition.onnx")

    threads = []

    for machine_id, cam in enumerate(cameras):
        thread = threading.Thread(
            target=identify,
            args=(sender, cam, detector, recognizer, machine_id),
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    gui_thread = threading.Thread(target=display_gui, daemon=True)
    gui_thread.start()

    try:
        for thread in threads:
            thread.join()
        gui_thread.join()
    except KeyboardInterrupt:
        print("\n🔴 KeyboardInterrupt detected! Exiting gracefully...")
        stop_event.set()
