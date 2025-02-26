import threading
import time

import cv2

from common.camera import Camera

# グローバル変数
recording = False
video_writer = None
frame_buffer = None  # フレームをメインスレッドに渡すための変数


def record_video():
    """録画処理 (録画のみ、GUI表示はしない)"""
    global recording, video_writer, frame_buffer

    cam = Camera(0, 640, 480)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter("output.mp4", fourcc, 20.0, (640, 480))

    while recording:
        frame = cam.get_frame()
        if frame is not None:
            video_writer.write(frame)
            frame_buffer = frame  # 最新のフレームをメインスレッドへ
        else:
            print("⚠️ フレーム取得失敗")

    cam.release()
    video_writer.release()
    print("📼 録画終了")


def show_gui():
    """GUI表示用の関数 (メインスレッドで動作)"""
    global frame_buffer
    while True:
        if frame_buffer is not None:
            cv2.imshow("Recording", frame_buffer)
        if cv2.waitKey(1) & 0xFF == ord("q"):  # `q` で強制停止
            break
    cv2.destroyAllWindows()


def cli_interface():
    """CLI の入力を処理"""
    global recording, video_writer

    gui_thread = threading.Thread(target=show_gui, daemon=True)
    gui_thread.start()

    while True:
        command = input("Enter command (rec_on / rec_off / exit): ").strip().lower()
        if command == "rec_on":
            if not recording:
                recording = True
                threading.Thread(target=record_video, daemon=True).start()
                print("🎥 録画開始 (MP4形式, GUI表示)")
            else:
                print("⚠️ すでに録画中です")
        elif command == "rec_off":
            if recording:
                recording = False
                time.sleep(1)  # スレッドが完全に終了するのを待つ
                print("⏹️ 録画停止")
            else:
                print("⚠️ 録画は開始されていません")
        elif command == "exit":
            if recording:
                recording = False
                time.sleep(1)  # スレッドが完全に終了するのを待つ
            break
        else:
            print("❌ 無効なコマンドです: rec_on / rec_off / exit を入力してください")

    cv2.destroyAllWindows()  # プログラム終了時にウィンドウを閉じる


if __name__ == "__main__":
    cli_interface()
