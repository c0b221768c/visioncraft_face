from config import config
from insightface.app import FaceAnalysis


class FaceDetector:
    def __init__(self):
        self.device = config.DEVICE
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider"]
            if self.device == "cuda"
            else ["CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0 if self.device == "cuda" else -1)

    def detect_face(self, frame):
        """
        顔を検出し、最も大きい顔のバウンディングボックスを返す。

        :param frame: 画像 (numpy array)
        :return: 顔の座標 [x1, y1, x2, y2] または None（検出失敗時）
        """
        faces = self.app.get(frame)

        if not faces:
            return None

        # 一番大きい顔を選択
        largest_face = max(
            faces,
            key=lambda face: (face.bbox[2] - face.bbox[0])
            * (face.bbox[3] - face.bbox[1]),
        )
        x1, y1, x2, y2 = map(int, largest_face.bbox)

        # 顔の座標が画像範囲内にあるかチェック
        h, w, _ = frame.shape
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            return None

        return [x1, y1, x2, y2]
