# identify_core.py

import threading
import time
from collections import Counter, deque

import cv2

from common.config import config
from common.utils import is_uuid, search_feature

lock = threading.Lock()  # スレッド間の排他制御用ロック

# グローバル変数（スレッド間で共有）
active_camera_id = None  # 現在アクティブなカメラ
active_uuid = None  # 現在認識されているユーザー
last_detected_time = None  # 最後に認識された時間


def identify(
    sender, camera, detector, recognizer, machine_id, frame_queues=None, send_data=True
):
    """
    各カメラ（machine_idごと）に識別処理を実行
    """
    global active_camera_id, active_uuid, last_detected_time

    print(f"🎥 Camera {machine_id} Started | Send Data: {send_data}")

    id_buffer = deque(maxlen=config.BUFFER_SIZE)
    frame_count = 0  # フレームカウント

    while True:
        frame = camera.get_frame()
        if frame is None:
            continue

        frame_count += 1
        if frame_count % config.FRAME_SKIP != 0:
            continue  # 設定したフレーム間隔でのみ処理

        faces = detector.detect_face(frame)
        if not faces:
            if frame_queues:
                frame_queues[machine_id].queue.clear()
                frame_queues[machine_id].put(frame)
            continue

        x1, y1, x2, y2 = faces  # 修正: `faces[0]` → `faces`

        if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
            continue

        face_crop = frame[y1:y2, x1:x2]

        if face_crop is None or face_crop.size == 0:
            continue

        feature = recognizer.extract_feature(face_crop)

        # Qdrant に問い合わせて検索
        best_match_uuid, best_match_sim = search_feature(feature)

        if best_match_sim < config.THRESHOLD:
            best_match_uuid = "Unknown"

        id_buffer.append(best_match_uuid)

        with lock:
            current_time = time.time()

            if len(id_buffer) == config.BUFFER_SIZE:
                most_common_id, count = Counter(id_buffer).most_common(1)[0]
                detected_uuid = (
                    most_common_id if count >= config.BUFFER_SIZE * 0.7 else "uncertain"
                )

                # **UUID検出ログ**
                print(f"🛠 Debug: Camera {machine_id} Detected: {detected_uuid}")

                # **送信条件の判定（send_data フラグを考慮）**
                if send_data and detected_uuid and is_uuid(detected_uuid):
                    if active_uuid is None or (
                        active_uuid == detected_uuid and active_camera_id != machine_id
                    ):
                        active_uuid = detected_uuid
                        active_camera_id = machine_id
                        last_detected_time = current_time
                        sender.send_request(machine_id=machine_id, uuid=detected_uuid)
                        print(f"✅ Sent from Camera {machine_id}: {detected_uuid}")

                # **一定時間経過でリセット**
                if active_uuid and current_time - last_detected_time > config.TIMEOUT:
                    print("🕒 Timeout: Reset active user")
                    active_uuid = None
                    active_camera_id = None

        # GUI用のフレーム保存
        if frame_queues:
            label = f"ID: {best_match_uuid} (Sim: {best_match_sim:.2f})"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            frame_queues[machine_id].queue.clear()
            frame_queues[machine_id].put(frame)

        time.sleep(0.1)
