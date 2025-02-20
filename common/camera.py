import time

import cv2


class Camera:
    def __init__(
        self, camera_index=0, width=640, height=480, retries=3, retry_delay=1.0
    ):
        """
        カメラクラス

        :param camera_index: 使用するカメラのインデックス
        :param width: キャプチャーする画像の幅
        :param height: キャプチャーする画像の高さ
        :param retries: カメラ接続のリトライ回数
        :param retry_delay: リトライ間隔（秒）
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.retries = retries
        self.retry_delay = retry_delay
        self.cap = None
        self._initialize_camera()

    def _initialize_camera(self):
        """カメラを初期化し、指定回数リトライする"""
        for attempt in range(self.retries):
            self.cap = cv2.VideoCapture(self.camera_index)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                return
            else:
                print(
                    f"⚠️ カメラ {self.camera_index} を開けません (試行 {attempt + 1}/{self.retries})"
                )
                time.sleep(self.retry_delay)

        raise RuntimeError(f"❌ カメラ {self.camera_index} を開けませんでした。")

    def get_frame(self):
        """
        フレームを取得する

        :return: 読み込んだフレーム (numpy array) or None
        """
        if self.cap is None or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        """カメラを解放する"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print(f"📷 カメラ {self.camera_index} を解放しました。")

    def __enter__(self):
        """コンテキストマネージャ用"""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """コンテキストマネージャ終了時にカメラを解放"""
        self.release()


if __name__ == "__main__":
    with Camera(0) as cam:
        for _ in range(10):
            frame = cam.get_frame()
            if frame is not None:
                cv2.imshow("Camera", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                print("⚠️ フレーム取得失敗")
        cv2.destroyAllWindows()
