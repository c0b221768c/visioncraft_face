import time

import cv2

from api.sender import SenderTCP
from common.camera import Camera
from common.config import config
from common.detection import FaceDetector

# 設定
timeout_active = False  # タイムアウト状態
timeout_start_time = None  # タイムアウト開始時刻
face_persist_time = None  # 顔が認識された開始時刻
running = True  # プログラムの実行フラグ
first_sended = False  # 初回送信フラグ


def get_elapsed_time(face_size, current_time):
    """
    顔の継続時間を計算する
    """
    global face_persist_time
    if face_size > config.MIN_FACE_SIZE:
        if face_persist_time is None:
            face_persist_time = current_time
            return 0.0  # 初回検出
        else:
            elapsed_time = current_time - face_persist_time
            return round(elapsed_time, 1)  # 小数点1桁まで丸める
    else:
        face_persist_time = None  # 顔が小さくなったらリセット
        return 0.0


# 初期化
sender = SenderTCP()
camera = Camera(0)  # カメラ1台のみ
detector = FaceDetector()

print("🎥 Camera started | Press 'ESC' to exit")

while running:
    frame = camera.get_frame()
    if frame is None:
        continue

    current_time = time.time()  # ループ内で変動しないようにする

    # 顔検出
    face = detector.detect_face(frame)
    if not face:
        face_persist_time = None  # 顔が見えなくなったらリセット
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            running = False
        continue

    # 顔のサイズ計算
    x1, y1, x2, y2 = face
    face_size = (x2 - x1) * (y2 - y1)

    elapsed_time = get_elapsed_time(face_size, current_time)

    print(f"Elapsed Time: {elapsed_time} sec")  # デバッグ用ログ

    if elapsed_time >= config.FACE_PERSIST_DURATION:
        if not first_sended:
            sender.send_request("dummy_uuid", 1)
            print("📡 Data sent for User")
            first_sended = True

    # # データ送信のタイミング
    # if (
    #     not timeout_active
    #     and face_size > config.MIN_FACE_SIZE
    #     and elapsed_time >= config.FACE_PERSIST_DURATION
    # ):
    #     sender.send_request("dummy_uuid", 1)
    #     print(f"📡 Data sent for User at {time.strftime('%H:%M:%S')}")
    #     timeout_active = True
    #     timeout_start_time = current_time

    # タイムアウト解除
    if timeout_active and current_time - timeout_start_time >= config.TIMEOUT_DURATION:
        timeout_active = False  # タイムアウト解除

    # 顔を枠で囲む
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    text = f"Face Size: {face_size}"
    cv2.putText(
        frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
    )

    # 映像を表示
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        running = False

# 終了処理
camera.release()
cv2.destroyAllWindows()
print("✅ Program terminated successfully.")
