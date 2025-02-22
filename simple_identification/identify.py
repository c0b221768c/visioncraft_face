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

# 初期化
sender = SenderTCP()
camera = Camera(0)  # カメラ1台のみ
detector = FaceDetector()

print("🎥 Camera started | Press 'ESC' to exit")

while running:
    frame = camera.get_frame()
    if frame is None:
        continue

    current_time = time.time()

    # 顔検出
    face = detector.detect_face(frame)
    if not face:
        print("⚠️ No face detected.")
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            running = False
        continue

    # 顔のサイズ計算
    x1, y1, x2, y2 = face
    face_size = (x2 - x1) * (y2 - y1)
    detected_long_enough = False

    # 顔が一定サイズを超えたか
    if face_size > config.MIN_FACE_SIZE:
        if face_persist_time is None:
            face_persist_time = current_time  # 初回検出時刻を記録
        elif current_time - face_persist_time >= config.FACE_PERSIST_DURATION:
            detected_long_enough = True
    else:
        face_persist_time = None  # 小さくなったらリセット

    # データ送信のタイミング
    if not timeout_active and face_size > config.MIN_FACE_SIZE and detected_long_enough:
        sender.send_request("dummy_uuid", 0)
        print(f"📡 Data sent for User at {time.strftime('%H:%M:%S')}")
        timeout_active = True
        timeout_start_time = current_time

    # タイムアウト解除
    if timeout_active and current_time - timeout_start_time >= config.TIMEOUT_DURATION:
        timeout_active = False  # タイムアウト解除

    # 顔を枠で囲む
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    text = f"Face Size: {face_size}"
    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 映像を表示
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        running = False

# 終了処理
camera.release()
cv2.destroyAllWindows()
print("✅ Program terminated successfully.")
