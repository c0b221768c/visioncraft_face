import time

import cv2

from config import config

class Camera:
    def __init__(self, index):
        self.index = index
        self.height = config.FRAME_HEIGHT
        self.width = config.FRAME_WIDTH
        self.cap = None
        self._init_camera()
    
    def _init_camera(self):
        self.cap = cv2.VideoCapture(self.index)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            return
        else:
            raise RuntimeError(f"❌ カメラ {self.index} を開けませんでした。")
        

    def get_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print(f"📷 カメラ {self.index} を解放しました。")