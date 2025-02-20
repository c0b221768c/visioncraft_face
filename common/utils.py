import signal
import uuid

# `KeyboardInterrupt` を無視する（インポート中に Ctrl + C を無視）
signal.signal(signal.SIGINT, signal.SIG_IGN)

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "face_features"

client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)

# `KeyboardInterrupt` を再度有効化
signal.signal(signal.SIGINT, signal.default_int_handler)


def save_feature(user_uuid, feature):
    """Qdrant に特徴量を保存"""
    feature = feature.flatten().tolist()
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[PointStruct(id=user_uuid, vector=feature, payload={"uuid": user_uuid})],
    )


def search_feature(feature, top_k=1):
    """Qdrant で類似特徴検索 (cosine_similarity を使用)"""
    feature = feature.flatten().tolist()
    search_results = client.search(
        collection_name=COLLECTION_NAME, query_vector=feature, limit=top_k
    )
    if search_results:
        best_match = search_results[0]
        return best_match.payload["uuid"], best_match.score
    return "Unknown", 0.0


def is_uuid(value):
    """値がUUID形式かどうかを判定"""
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return False
