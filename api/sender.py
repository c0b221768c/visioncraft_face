import json
import socket


class SenderTCP:
    def __init__(self, server_ip: str, server_port: int, buffer_size: int = 4096):
        """
        TCPクライアントを初期化

        :param server_ip: サーバーのIPアドレス (MacBookのIP)
        :param server_port: サーバーのポート番号
        :param buffer_size: 受信バッファサイズ（デフォルト4096）
        """
        self.server_ip = server_ip
        self.server_port = server_port
        self.buffer_size = buffer_size

    def send_request(self, uuid: str, machine_id: int):
        """
        サーバーにUUIDとmachine_idを送信する

        :param uuid: 送信するアクターのUUID
        :param machine_id: 送信する機械のID (0~3)
        :return: サーバーからのレスポンス
        """
        if not (0 <= machine_id <= 3):
            raise ValueError("machine_id は 0~3 の範囲で指定してください。")

        data = {"actor": "player", "type": "attract", "machine_id": machine_id, "user_id":uuid}

        # JSONデータをエンコード
        send_str = json.dumps(data, separators=(",", ":"))

        # TCPソケットを作成
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            # サーバーに接続
            client.connect((self.server_ip, self.server_port))
            print(f"📡 Sending data to {self.server_ip}:{self.server_port} ...")

            # JSONデータを送信
            client.send(send_str.encode())

            # サーバーからのレスポンスを受信
            response = client.recv(self.buffer_size)
            print("✅ Server Response:", response.decode())

        except Exception as e:
            print("❌ Error:", e)

        finally:
            client.close()  # 接続を閉じる
