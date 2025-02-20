import os
import sys
from collections import Counter, deque

import cv2  # GUI表示用

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.camera import Camera
from common.detection import FaceDetector
from common.utils import search_feature, is_uuid
from common.recognition import FaceRecognition
from api.sender import SenderTCP

THRESHOLD = 0.6
BUFFER_SIZE = 10
FRAME_SKIP = 5  # 5フレームごとに処理を実行

def identify(sender: SenderTCP, camera: Camera, detector: FaceDetector, recognizer: FaceRecognition):
    print("O : Start")

    id_buffer = deque(maxlen=BUFFER_SIZE)
    last_sent_id = None  # 前回送信したUUIDを記録
    frame_count = 0  # フレームカウント

    while True:
        frame = camera.get_frame()
        if frame is None:
            continue

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue  # 設定したフレーム間隔でのみ処理

        faces = detector.detect_face(frame)
        if not faces:
            print("X : Can not detect")
            cv2.imshow("Face Identification", frame)  # GUI表示
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        x1, y1, x2, y2 = faces[0]

        # ================= 追加: 範囲チェック =================
        if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
            print("X : Face bounding box is out of frame bounds")
            continue

        face_crop = frame[y1:y2, x1:x2]

        # ================= 追加: None チェック =================
        if face_crop is None or face_crop.size == 0:
            print("X : Extracted face_crop is None or empty")
            continue

        feature = recognizer.extract_feature(face_crop)

        # Qdrant に問い合わせて検索
        best_match_uuid, best_match_sim = search_feature(feature)

        if best_match_sim < THRESHOLD:
            best_match_uuid = "Unknown"

        id_buffer.append(best_match_uuid)

        if len(id_buffer) == BUFFER_SIZE:
            most_common_id, count = Counter(id_buffer).most_common(1)[0]
            if count >= BUFFER_SIZE * 0.7:
                final_id = most_common_id
            else:
                final_id = "uncertain"

            # 以前送信したIDと異なる場合のみ送信
            if is_uuid(final_id) and final_id != last_sent_id:
                sender.send_request(machine_id=0, uuid=final_id)
                last_sent_id = final_id  # 最後に送信したIDを更新
                print(f"✅ Sent ID: {final_id}")

            print(f"Success: {final_id} ({count}/{BUFFER_SIZE})")

        # ======= GUI表示追加 =======
        label = f"ID: {best_match_uuid} (Sim: {best_match_sim:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Face Identification", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    camera = Camera(0, 640, 480)
    detector = FaceDetector()
    recognizer = FaceRecognition("models/face_recognition.onnx")
    sender = SenderTCP("172.16.103.17", 8080)

    identify(sender, camera, detector, recognizer)
