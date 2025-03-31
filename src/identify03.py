import time
import cv2
from api.sender import SenderTCP
from camera import Camera
from config import config
from detection import FaceDetector


class FaceDrawer:
    """検出した顔に枠を描画するクラス"""
    
    @staticmethod
    def draw_face(frame, face, is_large):
        """顔に枠を描画する"""
        x1, y1, x2, y2 = face
        color = (0, 255, 0) if is_large else (0, 0, 255)  # 緑: 大きい顔, 赤: 小さい顔
        thickness = 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


class FaceIdentification:
    """顔検出と識別の処理を管理するクラス"""
    
    def __init__(self, input_cindex: int, output_cindex: int):
        self.camera = Camera(input_cindex)
        self.detector = FaceDetector()
        self.sender = SenderTCP()
        self.face_persist_time = None
        self.output_cindex = output_cindex

    def process_frame(self, frame):
        """フレームから顔を検出し、条件を満たせばデータ送信"""
        face = self.detector.detect_face(frame)
        current_time = time.time()

        if face:
            x1, y1, x2, y2 = face
            face_size = (x2 - x1) * (y2 - y1)
            is_large = face_size > config.FACE_SIZE_THRESHOLD

            # 顔に枠を描画
            FaceDrawer.draw_face(frame, face, is_large)

            if is_large:
                if self.face_persist_time is None:
                    self.face_persist_time = current_time
                else:
                    elapsed_time = current_time - self.face_persist_time
                    if elapsed_time > config.FACE_PERSIST_THRESHOLD:
                        self.sender.send_request("attract", "hello", self.output_cindex)
                        print("📡 Data sent: 'hello'")
                        self.face_persist_time = None
            else:
                self.face_persist_time = None
        else:
            self.face_persist_time = None

    def run(self):
        """カメラのフレームを取得し続け、顔識別処理を実行"""
        print(f"🚀 Starting identification(Camera0{self.output_cindex})...")

        while True:
            frame = self.camera.get_frame()
            if frame is None:
                continue

            self.process_frame(frame)
            cv2.imshow("Camera02", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESCキーで終了
                break

        self.camera.release()
        cv2.destroyAllWindows()
        print("🛑 Stopped identification.")


if __name__ == "__main__":
    face_identifier = FaceIdentification(input_cindex=3, output_cindex=3)
    face_identifier.run()
