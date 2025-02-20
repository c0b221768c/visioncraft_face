import cv2
import numpy as np
import onnxruntime as ort


class FaceRecognition:
    def __init__(
        self, model_path, device="cuda" if ort.get_device() == "GPU" else "cpu"
    ):
        """
        顔認識モデルを初期化する

        :param model_path: ONNXモデルのパス
        :param device: 使用するデバイス ('cuda' or 'cpu')
        """
        self.device = device
        providers = (
            ["CUDAExecutionProvider"]
            if self.device == "cuda"
            else ["CPUExecutionProvider"]
        )

        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.feature_dim = self.session.get_outputs()[0].shape[1]
        except Exception as e:
            raise RuntimeError(f"❌ ONNXモデルのロードに失敗: {e}")

    def preprocess(self, face):
        """
        画像の前処理 (リサイズ、正規化、チャネル変換)

        :param face: 入力画像 (numpy array)
        :return: ONNXモデル用の前処理済みテンソル
        """
        if face is None or face.size == 0:
            print("⚠️ Warning: 無効な顔画像が渡されました")
            return None

        try:
            face = cv2.resize(face, (self.input_shape[2], self.input_shape[3]))
            face = face.astype(np.float32) / 255.0  # 正規化
            face = np.transpose(face, (2, 0, 1))  # チャネル変換
            face = np.expand_dims(face, axis=0)  # バッチ次元追加
            return face
        except Exception as e:
            print(f"⚠️ Preprocessエラー: {e}")
            return None

    def extract_feature(self, face):
        """
        顔の特徴ベクトルを抽出

        :param face: 顔画像 (numpy array)
        :return: 正規化された特徴ベクトル (numpy array)
        """
        input_tensor = self.preprocess(face)
        if input_tensor is None:
            return np.zeros(
                self.feature_dim, dtype=np.float32
            )  # エラー時はゼロベクトルを返す

        try:
            feature = self.session.run(None, {self.input_name: input_tensor})[0]
            return feature / np.linalg.norm(feature)  # 正規化
        except Exception as e:
            print(f"❌ ONNX推論エラー: {e}")
            return np.zeros(
                self.feature_dim, dtype=np.float32
            )  # 失敗時にゼロベクトルを返す
