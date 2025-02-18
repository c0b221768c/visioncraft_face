import cv2
import numpy as np
import onnxruntime as ort

class FaceRecognition:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider"]
        )

        # ONNX
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.feature_dim = self.session.get_outputs()[0].shape[1]
    
    def preprocess(self, face):
        face = cv2.resize(face, (self.input_shape[2], self.input_shape[3]))
        face = face.astype(np.float32) / 255.0
        face = np.transpose(face, (2, 0, 1))
        face = np.expand_dims(face, axis=0)
        return face
    
    def extract_feature(self, face):
        input_tensor = self.preprocess(face)
        feature = self.session.run(None, {self.input_name: input_tensor})[0]
        return feature / np.linalg.norm(feature)