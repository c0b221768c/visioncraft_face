import os
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

class Config:
    # Camera settings
    FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", 640))
    FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", 480))

    # Device settings
    DEVICE = os.getenv("DEVICE", "cuda" if os.getenv("USE_CUDA", "true").lower() == "true" else "cpu")

    # Server settings
    SERVER_IP = os.getenv("SERVER_IP", "127.0.0.1")
    SERVER_PORT = int(os.getenv("SERVER_PORT", 8080))
    BUFFER_SIZE = int(os.getenv("BUFFER_SIZE", 4096))

    # Identification settings
    FACE_SIZE_THRESHOLD = int(os.getenv("FACE_SIZE_THRESHOLD", 10000))
    FACE_PERSIST_THRESHOLD = int(os.getenv("FACE_PERSIST_THRESHOLD", 3)) # seconds

config = Config()