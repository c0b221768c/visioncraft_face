import os
import sys
import uuid

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.camera import Camera
from common.config import config  # 設定をインポート
from common.detection import FaceDetector
from common.recognition import FaceRecognition
from common.utils import save_feature


def collect_face_features(
    camera: Camera, detector: FaceDetector, recognizer: FaceRecognition
):
    """
    カメラから顔を検出し、特徴量を収集する

    :param camera: Camera インスタンス
    :param detector: FaceDetector インスタンス
    :param recognizer: FaceRecognition インスタンス
    :return: 収集された特徴量のリスト (list of numpy arrays)
    """
    collected_features = []

    print("📸 顔をカメラに向けてください。")
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue

        faces = detector.detect_face(frame)
        if not faces:
            cv2.imshow("Face Registration", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        x1, y1, x2, y2 = faces
        face_crop = frame[y1:y2, x1:x2]
        feature = recognizer.extract_feature(face_crop)

        if feature is not None:
            collected_features.append(feature)
            print(
                f"✅ 登録中... {len(collected_features)}/{config.NUM_FEATURES} 特徴量収集中"
            )

        if len(collected_features) >= config.NUM_FEATURES:
            break

    camera.release()
    cv2.destroyAllWindows()

    return collected_features


def register_face(
    user_uuid: str, camera: Camera, detector: FaceDetector, recognizer: FaceRecognition
):
    """
    ユーザーの顔特徴量を収集し、Qdrant に保存する

    :param user_uuid: ユーザーのUUID
    :param camera: Camera インスタンス
    :param detector: FaceDetector インスタンス
    :param recognizer: FaceRecognition インスタンス
    """
    collected_features = collect_face_features(camera, detector, recognizer)

    if not collected_features:
        print("❌ 登録失敗: 十分な特徴量を収集できませんでした。")
        return

    avg_feature = np.mean(collected_features, axis=0)
    save_feature(user_uuid, avg_feature)

    print(f"🎉 登録成功！ UUID={user_uuid}")


if __name__ == "__main__":
    camera = Camera(
        config.CAMERA_INDEX_FOR_RECOGNITION, config.FRAME_WIDTH, config.FRAME_HEIGHT
    )  # 設定からカメラ設定を取得
    detector = FaceDetector()
    recognizer = FaceRecognition(
        config.FACE_MODEL_PATH, device=config.DEVICE
    )  # モデルパスとデバイス設定を適用

    user_uuid = str(uuid.uuid4())
    register_face(user_uuid, camera, detector, recognizer)
