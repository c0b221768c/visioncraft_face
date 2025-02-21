import uuid

from config import config
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

client = QdrantClient(config.QDRANT_HOST, port=config.QDRANT_PORT)


def save_feature(user_uuid, feature):
    """Qdrant に特徴量を保存"""
    try:
        feature = feature.flatten().tolist()
        client.upsert(
            collection_name=config.COLLECTION_NAME,
            points=[
                PointStruct(id=user_uuid, vector=feature, payload={"uuid": user_uuid})
            ],
        )
        print(f"✅ Feature saved for UUID: {user_uuid}")
    except Exception as e:
        print(f"❌ Error saving feature: {e}")


def search_feature(feature, top_k=1):
    """Qdrant で類似特徴検索 (cosine_similarity を使用)"""
    try:
        feature = feature.flatten().tolist()
        search_results = client.search(
            collection_name=config.COLLECTION_NAME, query_vector=feature, limit=top_k
        )
        if search_results:
            best_match = search_results[0]
            return best_match.payload["uuid"], float(best_match.score)
    except Exception as e:
        print(f"❌ Error searching feature: {e}")

    return "Unknown", 0.0


def is_uuid(value):
    """値がUUID形式かどうかを判定"""
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return False
