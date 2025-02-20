from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import uuid

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "face_features"

client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)


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

def is_uuid(value: str) -> bool:
    """
    指定された文字列がUUID4であるかを判別する関数

    :param value: 判定する文字列
    :return: UUID4であればTrue、そうでなければFalse
    """
    try:
        uuid_obj = uuid.UUID(value, version=4)
        return str(uuid_obj) == value
    except ValueError:
        return False