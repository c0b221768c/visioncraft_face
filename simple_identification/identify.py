import time
import cv2

from api.sender import SenderTCP
from common.camera import Camera
from common.config import config
from common.detection import FaceDetector


class UserState:
    """ユーザーの状態を管理"""
    def __init__(self):
        self.present = False  # ユーザーがカメラ範囲内にいるか
        self.first_sended = False  # 最初のデータ送信が済んだか
        self.timeout_active = False  # タイムアウトが有効か
        self.timeout_start_time = None  # タイムアウト開始時刻
        self.face_persist_time = None  # ユーザーが認識された開始時刻


def get_face_elapsed_time(face_size, current_time, user_state):
    """ ユーザーが検出され続けている時間を計測 """
    if face_size > config.MIN_FACE_SIZE:
        if user_state.face_persist_time is None:
            user_state.face_persist_time = current_time
            return 0.0  # 初回検出
        else:
            elapsed = current_time - user_state.face_persist_time
            return min(round(elapsed, 1), config.FACE_PERSIST_DURATION)  # しきい値以上は固定
    else:
        user_state.face_persist_time = None
        return 0.0  # 顔が小さくなったらリセット


def update_user_presence(elapsed_time, user_state, sender):
    """ ユーザーの状態を更新し、初回データ送信を行う """
    if elapsed_time >= config.FACE_PERSIST_DURATION and not user_state.present:
        print("👤 User presence confirmed.")
        user_state.present = True  # ユーザーがいると記録
        sender.send_request("attract","hello", 1)  # 滞在開始時に "hello" を送信
        print("📡 Data sent: 'hello'")

    if user_state.present and not user_state.first_sended:
        user_state.first_sended = True  # 初回送信済みに設定


def handle_timeout(user_state, current_time, sender):
    """ ユーザーの一時的な消失を処理し、完全に離れたかを判定 """
    if user_state.present and not user_state.timeout_active:
        user_state.timeout_active = True
        user_state.timeout_start_time = current_time
        print("⏳ User temporarily disappeared, starting timeout...")

    if user_state.timeout_active and current_time - user_state.timeout_start_time >= config.TIMEOUT_DURATION:
        print("❌ User left the area.")
        user_state.present = False
        user_state.timeout_active = False
        user_state.first_sended = False  # ユーザーが離れたら初回送信フラグをリセット
        sender.send_request("leave","goodbye", 1)  # 離脱時に "goodbye" を送信
        print("📡 Data sent: 'goodbye'")


def detect_face(frame, detector, user_state, sender, current_time):
    """ 顔を検出し、ユーザーの滞在状況を判定 """
    face = detector.detect_face(frame)
    if not face:
        if user_state.present:
            handle_timeout(user_state, current_time, sender)
        return None, None

    if user_state.timeout_active:
        print("✅ User returned before timeout ended.")
        user_state.timeout_active = False

    x1, y1, x2, y2 = face
    face_size = (x2 - x1) * (y2 - y1)
    elapsed_time = get_face_elapsed_time(face_size, current_time, user_state)

    update_user_presence(elapsed_time, user_state, sender)

    return face, face_size


# 初期化
sender = SenderTCP()
camera = Camera(0)  # カメラ1台のみ
detector = FaceDetector()
user_state = UserState()  # ユーザー状態管理

print("🎥 Camera started | Press 'ESC' to exit")

while True:
    frame = camera.get_frame()
    if frame is None:
        continue

    # `config.GAME_STATUS` が `True` の間は何もしない
    if config.GAME_STATUS:
        print(config.GAME_STATUS)
        print("🎮 GAME IN PROGRESS - Detection Paused")

        user_state.present = True #
        user_state.timeout_active = False #
        user_state.timeout_start_time = None #
        user_state.face_persist_time = time.time()

        time.sleep(1)  # CPU負荷を抑えるため、1秒スリープ
        continue

    current_time = time.time()  # ループ内で変動しないようにする

    face, face_size = detect_face(frame, detector, user_state, sender, current_time)

    if face is not None:
        x1, y1, x2, y2 = face
        elapsed_display = f"{current_time - user_state.face_persist_time:.1f} sec" if user_state.face_persist_time else "0.0 sec"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"Face Size: {face_size} | Elapsed: {elapsed_display}"
        cv2.putText(
            frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 終了処理
camera.release()
cv2.destroyAllWindows()
print("✅ Program terminated successfully.")
