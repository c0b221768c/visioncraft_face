import time
import tkinter as tk

import cv2

from api.sender import SenderTCP
from common.camera import Camera
from common.config import config
from common.detection import FaceDetector

# 設定
machine_id = 0  # 初期値はカメラ0
sender = SenderTCP()
camera = Camera(machine_id)  # カメラのインスタンスを作成
detector = FaceDetector()  # 顔検出モデルのインスタンスを作成
face_persist_time = None  # 顔検出の開始時刻を記録する変数
attract_sent = False  # attractが送信されたかどうかのフラグ


# 離脱処理
def leave_action():
    global attract_sent
    sender.send_request("leave", "goodbye")  # 離脱イベントを送信
    print("📡 Leave signal sent")  # ログ出力
    attract_sent = False  # 離脱後に attract を再度許可


# カメラを切り替えて attract を送信する関数
def attract_action(cam_id):
    global machine_id, camera, attract_sent
    if attract_sent:  # 既にattractが送信されていたら無視
        print(f"⚠️ Attract already sent for Camera {machine_id}. Ignoring request.")
        return
    #
    machine_id = cam_id  # 選択されたカメラIDに変更
    camera.release()  # 既存のカメラを解放
    camera = Camera(machine_id)  # 新しいカメラを設定
    #
    sender.send_request("attract", "hello", machine_id)  # Attract信号を送信
    print(f"📡 Attract signal sent for Camera {machine_id}")
    #
    attract_sent = True  # フラグをセットして再送防止

def detct_face(frame, detector):
    face = detector.detect_face(frame)
    if not face:
        return None, None
    
    x1, y1, x2, y2 = face
    face_size = (x2-x1) * (y2-y1)
    return face, face_size



def start_capture():
    global machine_id, sender, camera, detector, face_persist_time, attract_sent
    print("🎥 Camera started | Press 'ESC' to exit")
    while True:
        frame = camera.get_frame()  # カメラからフレームを取得
        if frame is None:
            continue  # フレームが取得できなければスキップ

        current_time = time.time()  # 現在時刻を取得
        face, face_size = detct_face(frame=frame, detector=detector)
        if face == None:
            continue

        if face_size >= config.MIN_FACE_SIZE and not attract_sent:  # すでにattractが送信されていたら無視
            if face_persist_time is None:
                face_persist_time = current_time  # 初回検出時に時刻を記録
            elif current_time - face_persist_time >= config.FACE_PERSIST_DURATION:
                sender.send_request("attract", "hello", machine_id)  # ユーザーの存在を送信
                print("📡 Attract signal sent")  # ログ出力
                attract_sent = True  # attract 送信済みフラグをセット
                face_persist_time = None  # タイマーをリセット
        else:
            face_persist_time = None  # 顔が検出されなくなったらリセット

        cv2.imshow("Camera", frame)  # 映像を表示
        if cv2.waitKey(1) & 0xFF == 27:  # ESCキーで終了
            break

def stop_capture():
    print("capture stop")

# GUIボタン作成
root = tk.Tk()
root.geometry("400x200")  # ウィンドウサイズ設定
root.title("カメラ選択 & 離脱")  # ウィンドウタイトル設定

# カメラ切り替えボタン
tk.Button(root, text="物理カメラ（スタート）", command=lambda: start_capture()).pack(
    fill=tk.X
)
tk.Button(root, text="物理カメラ（ストップ）", command=lambda: stop_capture()).pack(
    fill=tk.X
)
tk.Button(root, text="カメラ0 (アトラクト)", command=lambda: attract_action(0)).pack(
    fill=tk.X
)
tk.Button(root, text="カメラ1 (アトラクト)", command=lambda: attract_action(1)).pack(
    fill=tk.X
)
tk.Button(root, text="カメラ2 (アトラクト)", command=lambda: attract_action(2)).pack(
    fill=tk.X
)
tk.Button(root, text="カメラ3 (アトラクト)", command=lambda: attract_action(3)).pack(
    fill=tk.X
)

# 離脱ボタン
tk.Button(root, text="離脱", command=leave_action).pack(fill=tk.X)

root.mainloop()  # Tkinterのイベントループを開始



# 終了処理
camera.release()  # カメラを解放
cv2.destroyAllWindows()  # OpenCVのウィンドウを閉じる
print("✅ Program terminated successfully.")  # 終了メッセージ
