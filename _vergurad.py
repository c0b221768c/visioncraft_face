from qdrant_client import QdrantClient

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "face_features"

client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)

def fetch_all_data():
    """Qdrant 内の全データを取得して詳細表示"""
    scroll_filter = None  # フィルタなしで全データ取得
    batch_size = 10  # 一度に取得するデータ数
    next_offset = None  # 初回取得時のオフセット
    results = []

    while True:
        scroll_result = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=scroll_filter,
            limit=batch_size,
            offset=next_offset,
        )

        points, next_offset = scroll_result
        results.extend(points)

        if next_offset is None:  # すべてのデータを取得したら終了
            break

    for point in results:
        print("---------------------------------------------------")
        print(f"🔹 ID: {point.id}")

        # ベクターデータの有無をチェック
        if point.vector is None:
            print("⚠️ No vector data found!")
        else:
            print(f"✅ Vector Size: {len(point.vector)}")

        # UUIDを取得
        if point.payload:
            uuid = point.payload.get("uuid", "Unknown")
            print(f"📌 UUID: {uuid}")
        else:
            print("⚠️ No payload found!")

        # 生データをすべて出力（デバッグ用）
        print("📝 Raw Data:", point)

if __name__ == "__main__":
    fetch_all_data()
