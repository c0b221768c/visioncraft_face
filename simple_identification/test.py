import time
import cv2
import multiprocessing
from multiprocessing import shared_memory
from api.sender import SenderTCP
from api.receiver import ReceiverProcess
from common.camera import Camera
from common.config import config
from common.detection import FaceDetector


class UserState:
    """状態管理"""
    def __init__(self):
        self.is_send = False  # 送信済みフラグ
        self.persist_start_time = None  # 顔検出が継続した開始時刻


class CameraProcess(multiprocessing.Process):
    def __init__(self, camera_id, num_cameras, shared_flags):
        """
        カメラプロセスを初期化
        :param camera_id: このカメラのID
        :param num_cameras: 全カメラ数
        :param shared_flags: 共有メモリの送信フラグ
        """
        super().__init__()
        self.camera_id = camera_id
        self.num_cameras = num_cameras
        self.shared_flags = shared_flags  # 共有メモリへの参照
        self.camera = Camera(camera_id)
        self.detector = FaceDetector()
        self.sender = SenderTCP()
        self.user_state = UserState()

    def run(self):
        """ カメラ処理を開始 """
        print(f"🎥 Camera {self.camera_id} started | Press 'ESC' to exit")

        while True:
            # 他のカメラのフラグをチェック
            if any(self.shared_flags[i] for i in range(self.num_cameras) if i != self.camera_id):
                # 他のカメラがゲーム中なら黒背景にする
                frame = self.create_black_screen()
            else:
                frame = self.camera.get_frame()
                if frame is None:
                    continue

                face, face_size = self.detect_face(frame)
                self.handle_face_state(face_size)

                # 顔がある場合、フレームに描画
                if face is not None:
                    x1, y1, x2, y2 = face
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = f"Face Size: {face_size}"
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

            cv2.imshow(f"Camera {self.camera_id}", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        # 終了処理
        self.camera.release()
        cv2.destroyAllWindows()

    def detect_face(self, frame):
        """ 顔を検出し、サイズを取得 """
        face = self.detector.detect_face(frame)
        if not face:
            return None, None

        x1, y1, x2, y2 = face
        face_size = (x2 - x1) * (y2 - y1)
        return face, face_size

    def handle_face_state(self, face_size):
        """ ユーザーの顔が検出された際の処理 """
        current_time = time.time()

        if face_size is not None and face_size > config.MIN_FACE_SIZE:
            if self.user_state.persist_start_time is None:
                self.user_state.persist_start_time = current_time  # 初回検出時間を記録
            elif (current_time - self.user_state.persist_start_time) >= config.FACE_PERSIST_DURATION:
                if not self.user_state.is_send:
                    print(f"👤 Camera {self.camera_id}: User detected! Sending request...")
                    self.sender.send_request("attract", "hello", self.camera_id)
                    self.shared_flags[self.camera_id] = 1  # 送信フラグを立てる
                    self.user_state.is_send = True
        else:
            self.user_state.persist_start_time = None

        # `receiver.py` から `Action.WAIT` を受信したらリセット
        response = self.sender.send_request("status", "", self.camera_id)
        if response and "type" in response and response["type"] == "wait":
            print(f"🔄 Camera {self.camera_id}: Resetting user state...")
            self.shared_flags[self.camera_id] = 0  # 送信フラグをリセット
            self.user_state.is_send = False
            self.user_state.persist_start_time = None  # 経過時間リセット

    def create_black_screen(self):
        """ 他のカメラがゲーム中のときの黒背景画面を作成 """
        frame = 255 * (cv2.cvtColor(cv2.imread(config.BLACK_SCREEN_PATH), cv2.COLOR_BGR2GRAY) > 0).astype("uint8")
        cv2.putText(frame, "Another camera is playing", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return frame


# 設定
NUM_CAMERAS = 4  # カメラの台数

# 共有メモリの作成（各カメラの送信フラグを共有）
shared_mem = shared_memory.SharedMemory(create=True, size=NUM_CAMERAS)
shared_flags = memoryview(shared_mem.buf).cast('B')

# カメラプロセスを作成（複数台）
cameras = [CameraProcess(i, NUM_CAMERAS, shared_flags) for i in range(NUM_CAMERAS)]
receiver = ReceiverProcess(shared_flags)  # `receiver.py` もフラグを参照

# すべてのプロセスを開始
receiver.start()
for cam in cameras:
    cam.start()

# 終了処理
receiver.join()
for cam in cameras:
    cam.join()
