from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Qdrant サーバーの設定
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "face_features"

# クライアントインスタンスの作成
client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)


def init_qdrant_collection():
    """
    Qdrant に `face_features` コレクションを作成する
    (既に存在する場合はスキップ)
    """
    existing_collections = client.get_collections()

    if COLLECTION_NAME in [col.name for col in existing_collections.collections]:
        print(f"✅ コレクション '{COLLECTION_NAME}' は既に存在します。スキップします。")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )
    print(f"🎉 コレクション '{COLLECTION_NAME}' を作成しました！")


if __name__ == "__main__":
    print("🛠 Qdrant データベースの初期化を開始します...")
    init_qdrant_collection()
    print("✅ Qdrant データベースの初期化が完了しました。")
