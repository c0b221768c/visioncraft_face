import os
import sys
import cv2
import time
import threading
import queue
from collections import Counter, deque
import signal

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.camera import Camera
from common.detection import FaceDetector
from common.utils import search_feature, is_uuid
from common.recognition import FaceRecognition
from api.sender import SenderTCP

THRESHOLD = 0.6
BUFFER_SIZE = 10
FRAME_SKIP = 10  # 5フレームごとに処理を実行

# GUI表示用のフレームを保存するキュー（各カメラ用）
frame_queues = {i: queue.Queue(maxsize=1) for i in range(4)}

# 停止イベント（`Ctrl + C` を検出してすべてのスレッドを終了する）
stop_event = threading.Event()


def identify(sender: SenderTCP, camera: Camera, detector: FaceDetector, recognizer: FaceRecognition, machine_id: int):
    """
    各カメラ（machine_idごと）に識別処理を実行
    """
    print(f"🎥 Camera {machine_id} Started")

    id_buffer = deque(maxlen=BUFFER_SIZE)
    last_sent_id = None  # 各カメラごとに送信済みのIDを管理
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
            frame_queues[machine_id].queue.clear()  # キューをクリア
            frame_queues[machine_id].put(frame)  # GUI用に送信
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

        if len(id_buffer) == BUFFER_SIZE:
            most_common_id, count = Counter(id_buffer).most_common(1)[0]
            if count >= BUFFER_SIZE * 0.7:
                final_id = most_common_id
            else:
                final_id = "uncertain"

            if is_uuid(final_id) and final_id != last_sent_id:
                sender.send_request(machine_id=machine_id, uuid=final_id)
                last_sent_id = final_id  # 最後に送信したIDを更新
                print(f"✅ Sent from Camera {machine_id}: {final_id}")

        # フレームをGUI用のキューに送信
        label = f"ID: {best_match_uuid} (Sim: {best_match_sim:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        frame_queues[machine_id].queue.clear()
        frame_queues[machine_id].put(frame)

        time.sleep(0.1)  # CPU負荷を抑えるための短いスリープ


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


# ================================================================
# 🎯 **マルチスレッドで4台のカメラを同時に動作させる**
# ================================================================
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)  # `Ctrl + C` をキャッチ

    CAMERA01 = Camera(0, 640, 480)
    CAMERA02 = Camera(1, 640, 480)
    # CAMERA03 = Camera(2, 640, 480)
    # CAMERA04 = Camera(3, 640, 480)
    # cameras = [CAMERA01]
    cameras = [CAMERA01, CAMERA02]
    # cameras = [CAMERA01, CAMERA02, CAMERA03]
    # cameras = [CAMERA01, CAMERA02, CAMERA03, CAMERA04]

    sender = SenderTCP("172.27.112.1", 8080)
    detector = FaceDetector()
    recognizer = FaceRecognition("models/face_recognition.onnx")

    threads = []

    # 識別スレッドを開始
    for machine_id, cam in enumerate(cameras):
        thread = threading.Thread(target=identify, args=(sender, cam, detector, recognizer, machine_id), daemon=True)
        thread.start()
        threads.append(thread)

    # GUIスレッドを開始
    gui_thread = threading.Thread(target=display_gui, daemon=True)
    gui_thread.start()

    # すべてのスレッドが終了するまで待機
    try:
        for thread in threads:
            thread.join()
        gui_thread.join()
    except KeyboardInterrupt:
        print("\n🔴 KeyboardInterrupt detected! Exiting gracefully...")
        stop_event.set()
