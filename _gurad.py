import cv2
import numpy as np

from common.camera import Camera
from common.detection import FaceDetector
from common.recognition import FaceRecognition


def test_face_recognition():
    """カメラ映像から顔を検出し、特徴量を抽出するテスト"""

    # カメラ、顔検出、顔認識のセットアップ
    camera = Camera(0, 640, 480)
    detector = FaceDetector()
    recognizer = FaceRecognition("models/face_recognition.onnx")

    print(
        "🎥 カメラを起動しました。顔認識テストを開始します (終了するには 'q' を押してください)。"
    )

    while True:
        frame = camera.get_frame()
        if frame is None:
            print("⚠️ フレーム取得失敗")
            continue

        face_bbox = detector.detect_face(frame)

        # 顔が検出されなかった場合
        if face_bbox is None:
            cv2.imshow("Face Recognition Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # 顔の座標取得
        x1, y1, x2, y2 = face_bbox  # ここを修正

        face_crop = frame[y1:y2, x1:x2]

        # 特徴量を抽出
        feature = recognizer.extract_feature(face_crop)

        if feature is not None and np.linalg.norm(feature) > 0:
            label = "Face Recognized"
            color = (0, 255, 0)  # 緑
        else:
            label = "Recognition Failed"
            color = (0, 0, 255)  # 赤

        # 検出した顔の枠とテキストを描画
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

        cv2.imshow("Face Recognition Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    print("🛑 テストを終了しました。")


if __name__ == "__main__":
    test_face_recognition()
