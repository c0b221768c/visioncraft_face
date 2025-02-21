from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from common.config import config

# クライアントインスタンスの作成
client = QdrantClient(config.QDRANT_HOST, port=config.QDRANT_PORT)


def init_qdrant_collection():
    """
    Qdrant に `face_features` コレクションを作成する
    (既に存在する場合はスキップ)
    """
    existing_collections = client.get_collections()

    if config.COLLECTION_NAME in [col.name for col in existing_collections.collections]:
        print(
            f"✅ コレクション '{config.COLLECTION_NAME}' は既に存在します。スキップします。"
        )
        return

    client.create_collection(
        collection_name=config.COLLECTION_NAME,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )
    print(f"🎉 コレクション '{config.COLLECTION_NAME}' を作成しました！")


if __name__ == "__main__":
    print("🛠 Qdrant データベースの初期化を開始します...")
    init_qdrant_collection()
    print("✅ Qdrant データベースの初期化が完了しました。")
