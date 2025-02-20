# gui.py

import cv2
from config import NUM_CAMERAS


def display_gui(frame_queues, stop_event):
    """
    GUIを表示し、カメラのフレームをリアルタイムで描画
    """
    try:
        while not stop_event.is_set():
            for machine_id in range(NUM_CAMERAS):
                if not frame_queues[machine_id].empty():
                    frame = frame_queues[machine_id].get()
                    if frame is not None:
                        cv2.imshow(f"Camera {machine_id}", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):  # `Q` キーで正常終了
                print("\n🔴 Qキーが押されました。終了します...")
                stop_event.set()
                break
    finally:
        cv2.destroyAllWindows()
