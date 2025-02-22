import os
from pathlib import Path

from dotenv import load_dotenv

# `common/.env` の絶対パスを取得してロード
dotenv_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path)


class Config:
    """プロジェクトの設定管理"""

    # Qdrant 設定
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "face_features")

    # カメラ設定
    FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", 640))
    FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", 480))
    NUM_CAMERAS = int(os.getenv("NUM_CAMERAS", 1))

    # モデル設定
    FACE_MODEL_PATH = os.getenv("FACE_MODEL_PATH", "models/face_recognition.onnx")
    DEVICE = os.getenv(
        "DEVICE", "cuda" if os.getenv("USE_CUDA", "true").lower() == "true" else "cpu"
    )

    # サーバー設定
    SERVER_IP = os.getenv("SERVER_IP", "127.0.0.1")
    SERVER_PORT = int(os.getenv("SERVER_PORT", 8080))

    # identificationで使用
    THRESHOLD = float(os.getenv("THRESHOLD", 0.6))
    BUFFER_SIZE = int(os.getenv("BUFFER_SIZE", 10))
    FRAME_SKIP = int(os.getenv("FRAME_SKIP", 1))
    TIMEOUT = int(os.getenv("TIMEOUT", 10))

    # registrationで使用
    CAMERA_INDEX_FOR_RECOGNITION = int(os.getenv("CAMERA_INDEX", 0))
    NUM_FEATURES = int(os.getenv("NUM_FEATURES", 10))

    MIN_FACE_SIZE = int(os.getenv("MIN_FACE_SIZE", 10000))
    TIMEOUT_DURATION = int(os.getenv("TIMEOUT_FACE_SIZE", 10))
    FACE_PERSIST_DURATION = int(os.getenv("FACE_PERSIST_DURATION", 3))


config = Config()
