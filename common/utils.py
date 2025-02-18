import sqlite3
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "..", "data", "face_recognition.db")

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT UNIQUE NOT NULL,
                feature BLOB NOT NULL
              )
''')
    conn.commit()
    conn.close()

def save_feature(uuid, feature):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    feature_blob = feature.tobytes()
    c.execute("INSERT INTO features (uuid, feature) VALUES (?,?)", (uuid, feature_blob))
    conn.commit()
    conn.close()

def load_all_features():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT uuid, feature FROM features")
    data = c.fetchall()
    conn.close()

    uuids = []
    features = []
    for uuid, feature_blob in data:
        uuids.append(uuid)
        features.append(np.frombuffer(feature_blob, dtype=np.float32))
    return uuids, np.array(features)

def cosine_similarity(feature1, feature2):
    dot_product = np.dot(feature1, feature2)
    norm1 = np.linalg.norm(feature1)
    norm2 = np.linalg.norm(feature2)
    return dot_product / (norm1 * norm2)