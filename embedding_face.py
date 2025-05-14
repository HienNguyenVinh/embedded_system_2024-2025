import os
import cv2
import pickle
import tflite_runtime.interpreter as tf
import numpy as np
from settings import *
import io
from face_utils import init_face_recognition_model
from face_detector import FaceDetector

face_detector = FaceDetector()
interpreter = init_face_recognition_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_dtype = input_details[0]['dtype']


# --- Hàm tiền xử lý và embedding ---
def preprocess_and_embed(face_image, interpreter, input_details, output_details, input_dtype):
    """Resize, chuẩn hóa và tạo embedding bằng TFLite model."""
    try:
        resized_face = cv2.resize(face_image, (INPUT_WIDTH, INPUT_HEIGHT))
        rgb_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
        normalized_face = (rgb_face.astype(np.float32) - NORM_MEAN) / NORM_STD
        input_data = np.expand_dims(normalized_face, axis=0)

        if input_data.dtype != input_dtype:
             input_data = input_data.astype(input_dtype)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        embedding = interpreter.get_tensor(output_details[0]['index'])

        return embedding[0]

    except Exception as e:
        # print(f"Error in preprocess/embed realtime: {e}")
        return None


def get_face_embedding_tflite(image, face_detector, interpreter):
    """Nhận diện khuôn mặt và tạo embedding từ ảnh đầu vào."""
    if image is None:
        print("Error: Cannot read image.")
        return None
    
    resized_image = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))
    face_results = face_detector.get_faces(resized_image)

    if not face_results:
        print("Error: No faces detected.")
        return None

    best_crop, _ = max(face_results, key=lambda item: item[1][2] * item[1][3])

    embedding = preprocess_and_embed(best_crop, interpreter, 
                                    interpreter.get_input_details(), 
                                    interpreter.get_output_details(), 
                                    interpreter.get_input_details()[0]['dtype'])
    if embedding is None:
        print("Error: Failed to create embedding.")
        return None

    return embedding

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
                print(f"Đã tải {len(known_face_ids)} embeddings TFLite đã biết.")

            return known_face_embeddings, known_face_ids

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file embeddings '{EMBEDDINGS_FILE}'.")
        # exit()
    except Exception as e:
        print(f"Lỗi khi tải file embeddings: {e}")
        # exit()



# --- Main (Giữ nguyên logic chính, chỉ thay đổi hàm gọi) ---
if __name__ == "__main__":
    known_face_embeddings = []
    known_face_ids = []

    print(f"Bắt đầu quét ảnh trong thư mục: {KNOWN_FACES_DIR}")
    if not os.path.exists(KNOWN_FACES_DIR):
        print(f"Lỗi: Thư mục {KNOWN_FACES_DIR} không tồn tại.")
        exit()

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    image_files = [f for f in os.listdir(KNOWN_FACES_DIR) if os.path.isfile(os.path.join(KNOWN_FACES_DIR, f))]

    if not image_files:
        print(f"Không tìm thấy file ảnh nào trong {KNOWN_FACES_DIR}.")
        exit()

    for image_file in image_files:
        image_path = os.path.join(KNOWN_FACES_DIR, image_file)
        username = os.path.splitext(image_file)[0]
        print(f"Đang xử lý ảnh: {image_file} cho người dùng: {username}")
        image = cv2.imread(image_path)

        if image is None:
            print(f"Không thể đọc file ảnh: {image_path}")
            continue

        # Gọi hàm mới sử dụng TFLite
        embedding = get_face_embedding_tflite(image, face_detector, interpreter)

        if embedding is not None:
            known_face_embeddings.append(embedding)
            known_face_ids.append(username)
            print(f"-> Đã tạo embedding TFLite thành công cho {username}.")
        else:
            print(f"-> Không thể tạo embedding TFLite cho {username} từ ảnh {image_file}.")

    if not known_face_embeddings:
        print("Không tạo được embedding nào. Vui lòng kiểm tra lại ảnh đầu vào và model TFLite.")
        exit()

    # Lưu embeddings và tên vào file pickle (dùng tên file mới)
    data = {"embeddings": known_face_embeddings, "ids": known_face_ids}
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(data, f)

    print(f"\nHoàn tất! Đã lưu {len(known_face_embeddings)} embeddings TFLite vào file: {EMBEDDINGS_FILE}")