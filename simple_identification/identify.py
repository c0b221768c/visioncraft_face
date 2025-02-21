import threading
import time

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
            continue

        x1, y1, x2, y2 = face

        face_size = (x2 - x1) * (y2 - y1)  # 顔のサイズ計算
        current_time = time.time()

        with lock:
            if face_size > config.MIN_FACE_SIZE:
                # 継続時間を記録
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
                # 顔が小さくなった場合は継続時間リセット
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
