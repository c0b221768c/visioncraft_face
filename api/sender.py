import requests
import json

class Sender:
    def __init__(self, ip: str, port: int):
        """IPアドレスとポートを設定"""
        self.base_url = f"http://{ip}:{port}"

    def send_request(self, uuid: str, machine_id: int):
        """
        指定した UUID と machine_id を JSON 形式で送信

        :param uuid: 送信するアクターのUUID
        :param machine_id: 送信する機械のID (0~3)
        :return: APIレスポンス
        """
        if not (0 <= machine_id <= 3):
            raise ValueError("machine_id は 0~3 の範囲で指定してください。")

        data = {
            "actor": uuid,
            "type": "attract",
            "machine_id": machine_id
        }

        url = f"{self.base_url}/api/receive"
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(url, data=json.dumps(data), headers=headers)
            response.raise_for_status()  # HTTPエラーがあれば例外発生
            return response.json()  # JSONレスポンスを返す
        except requests.exceptions.RequestException as e:
            print(f"❌ Error sending request: {e}")
            return None

# ======= 動作テスト用 =======
if __name__ == "__main__":
    sender = Sender("127.0.0.1", 8080)  # IPとポートを適宜変更
    response = sender.send_request("550e8400-e29b-41d4-a716-446655440000", 2)
    print("✅ Response:", response)
