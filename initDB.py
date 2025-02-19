from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "face_features"

client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)


def init_qdrant_collection():
    """Qdrant にコレクションを作成"""
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )
    print(f"✅ Collection '{COLLECTION_NAME}' created successfully!")


if __name__ == "__main__":
    init_qdrant_collection()
