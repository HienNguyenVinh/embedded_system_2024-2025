import pickle
from settings import EMBEDDINGS_FILE
import os

print("Loading embeddings from file...")

def get_embeddings_data():
    try:
        if not os.path.isfile(EMBEDDINGS_FILE):
            raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_FILE}")
        
        with open(EMBEDDINGS_FILE, "rb") as f:
            data = pickle.load(f)
            known_face_embeddings = data["embeddings"]
            known_face_ids = data["ids"]

            if len(known_face_embeddings) == 0 or len(known_face_ids) == 0:
                print("Cảnh báo: File embeddings trống hoặc không hợp lệ.")
                known_face_embeddings = []
                known_face_ids = []
            else:
                for user_id, embedding in zip(known_face_ids, known_face_embeddings):
                    print(f"ID: {user_id} - Embedding: {embedding}")

                print(f"Đã tải {len(known_face_ids)} embeddings TFLite đã biết.")

            return known_face_embeddings, known_face_ids

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file embeddings '{EMBEDDINGS_FILE}'.")
    except Exception as e:
        print(f"Lỗi khi tải file embeddings: {e}")

get_embeddings_data()
