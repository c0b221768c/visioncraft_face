import torch
from insightface.app import FaceAnalysis

class FaceDetector:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'] if self.device == "cuda" else ["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0 if self.device == "cuda" else -1)

    def detect_face(self, frame):
        faces = self.app.get(frame)

        if not faces:
            return []

        # 一番大きい顔を選択
        largest_face = max(faces, key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))
        x1, y1, x2, y2 = map(int, largest_face.bbox)

        return [[x1, y1, x2, y2]]  # 一番大きな顔のみを返す
