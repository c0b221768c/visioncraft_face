import socket
import json
import os

class ReceiverTCP:
    def __init__(self, host="127.0.0.1", port=3035, buffer_size=1024, env_path="./common/.env"):
        """
        ReceiverTCPクラス: 指定されたホストとポートでTCPサーバーを起動し、ゲーム状態を受信する。

        :param host: 受信するホスト (デフォルト: 127.0.0.1)
        :param port: 受信するポート (デフォルト: 3035)
        :param buffer_size: 受信バッファサイズ (デフォルト: 1024)
        :param env_path: .envファイルのパス (デフォルト: ./common/.env)
        """
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.env_path = env_path  # .env ファイルのパス

    def start_server(self):
        """ TCPサーバーを起動し、クライアントからの接続を待ち受ける """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind((self.host, self.port))
            server_socket.listen(5)  # 最大5クライアント接続待ち
            print(f"[*] ReceiverTCP started on {self.host}:{self.port}, waiting for connections...")

            while True:
                client_socket, addr = server_socket.accept()
                print(f"[*] Connected by {addr}")
                self.handle_client(client_socket)

    def handle_client(self, client_socket):
        """ クライアントからデータを受信し、処理する """
        with client_socket:
            try:
                data = client_socket.recv(self.buffer_size).decode('utf-8')  # バイトデータを文字列にデコード
                print(f"[*] Raw Data Received: {data}")  # デバッグ用にデータを出力

                if not data:
                    print("[!] No data received.")
                    return

                # JSONデコードを試みる
                received_json = json.loads(data)
                if not isinstance(received_json, dict):  # JSONが辞書型でない場合エラー
                    raise ValueError("Received JSON is not a dictionary.")

                # is_game_active の取得
                is_game_active = received_json.get("is_game_active")
                print(f"[*] Game Status Received: is_game_active = {is_game_active}")

                # 受信データに基づいて処理を実行
                self.process_game_status(is_game_active)

            except json.JSONDecodeError:
                print("[!] JSON Decode Error: Invalid JSON format.")
            except ValueError as e:
                print(f"[!] Value Error: {e}")

    def process_game_status(self, is_game_active):
        """ 受信したゲームの状態に応じて .env ファイルを書き換える """
        if is_game_active is True:
            print("[+] ゲームが開始されました！")
            self.update_env("GAME_STATUS", "ACTIVE")  # ゲーム開始
        elif is_game_active is False:
            print("[-] ゲームが終了しました。")
            self.update_env("GAME_STATUS", "INACTIVE")  # ゲーム終了
        else:
            print("[!] 受信データが正しくありません。")

    def update_env(self, key, value):
        """ .env ファイル内の指定キーの値を更新する """
        env_lines = []
        updated = False

        # .env ファイルを読み込む
        if os.path.exists(self.env_path):
            with open(self.env_path, "r", encoding="utf-8") as file:
                env_lines = file.readlines()

        # 既存のキーを更新
        for i, line in enumerate(env_lines):
            if line.startswith(f"{key}="):
                env_lines[i] = f"{key}={value}\n"
                updated = True
                break

        # キーが存在しなければ追加
        if not updated:
            env_lines.append(f"{key}={value}\n")

        # .env ファイルを書き換える
        with open(self.env_path, "w", encoding="utf-8") as file:
            file.writelines(env_lines)

        print(f"[*] .env updated: {key}={value}")

if __name__ == "__main__":
    receiver = ReceiverTCP()
    receiver.start_server()
