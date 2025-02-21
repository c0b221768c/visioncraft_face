import threading
import time

import cv2

from common.config import config

lock = threading.Lock()

active_camera_id = None  # 現在アクティブなカメラ
active_user_id = None  # 現在認識されているユーザー
last_detected_time = None  # 最後に認識された時間
timeout_active = False  # タイムアウト状態
timeout_start_time = None  # タイムアウト開始時刻
face_persist_time = {}  # 各カメラごとの継続時間記録


def identify(sender, camera, detector, machine_id, frame_queues=None, send_data=True):
    global \
        active_camera_id, \
        active_user_id, \
        last_detected_time, \
        timeout_active, \
        timeout_start_time

    print(f"🎥 Camera {machine_id} Started | Send Data: {send_data}")

    while True:
        frame = camera.get_frame()
        if frame is None:
            continue

        face = detector.detect_face(frame)
        if not face:
            if frame_queues:
                frame_queues[machine_id].queue.clear()
                frame_queues[machine_id].put(frame)
            cv2.imshow(f"Camera {machine_id}", frame)  # 検出なしでもウィンドウ表示
            if cv2.waitKey(1) & 0xFF == 27:
                print("🚪 ESCキーが押されたため終了します。")
                break
            continue

        x1, y1, x2, y2 = face
        face_crop = frame[y1:y2, x1:x2]

        if face_crop is None or face_crop.size == 0:
            continue

        face_size = (x2 - x1) * (y2 - y1)  # 顔のサイズ計算
        current_time = time.time()

        with lock:
            if face_size > config.MIN_FACE_SIZE:
                if machine_id not in face_persist_time:
                    face_persist_time[machine_id] = current_time
                elif (
                    current_time - face_persist_time[machine_id]
                    >= config.FACE_PERSIST_DURATION
                ):
                    detected_long_enough = True
                else:
                    detected_long_enough = False
            else:
                face_persist_time[machine_id] = None
                detected_long_enough = False

            # タイムアウトチェック
            if timeout_active:
                if current_time - timeout_start_time >= config.TIMEOUT_DURATION:
                    timeout_active = False  # タイムアウト解除
                else:
                    print(f"⏳ Timeout active for Camera {machine_id}, skipping...")
                    continue  # 送信せずスキップ

            # 送信条件 (顔サイズが一定以上 & N秒継続)
            if face_size > config.MIN_FACE_SIZE and detected_long_enough:
                active_camera_id = machine_id
                user_uuid = "dummy_uuid"  # 実際には認識処理を実装
                sender.send_request(user_uuid, machine_id)
                print(f"📡 Data sent for User {user_uuid} from Camera {machine_id}")

                # タイムアウト開始
                timeout_active = True
                timeout_start_time = current_time

        # --- ここから顔を枠で囲んでサイズを表示 ---
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 緑の枠
        text = f"Face Size: {face_size}"
        cv2.putText(
            frame,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow(f"Camera {machine_id}", frame)  # 更新して表示
        if cv2.waitKey(1) & 0xFF == 27:
            print("🚪 ESCキーが押されたため終了します。")
            break
