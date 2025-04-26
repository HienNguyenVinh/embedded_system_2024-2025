import os
import cv2
import pickle
import mediapipe as mp
import tensorflow as tf
import numpy as np
from settings import *
import io
from face_camera import init_face_recognition_model

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=DETECTION_CONFIDENCE)
interpreter = init_face_recognition_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_dtype = input_details[0]['dtype']


# --- Hàm tiền xử lý và embedding ---
def preprocess_and_embed(face_image):
    """Resize, chuẩn hóa và tạo embedding bằng TFLite model."""
    try:
        resized_face = cv2.resize(face_image, (INPUT_WIDTH, INPUT_HEIGHT))
        rgb_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
        normalized_face = (rgb_face.astype(np.float32) - NORM_MEAN) / NORM_STD
        input_data = np.expand_dims(normalized_face, axis=0)


        # Kiểm tra và chuyển đổi dtype nếu cần
        if input_data.dtype != input_dtype:
             input_data = input_data.astype(input_dtype)


        # Inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        embedding = interpreter.get_tensor(output_details[0]['index'])

        return embedding[0] # Bỏ chiều batch

    except Exception as e:
        print(f"Lỗi trong quá trình tiền xử lý hoặc embedding: {e}")
        return None


def get_face_embedding_tflite(image):
    """Phát hiện khuôn mặt lớn nhất và trích xuất embedding bằng TFLite."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)

    if not results.detections:
        # print("Không phát hiện thấy khuôn mặt nào.")
        return None

    best_detection = None
    max_area = 0
    ih, iw, _ = image.shape

    for detection in results.detections:
        try:
            bboxC = detection.location_data.relative_bounding_box
            if bboxC is None: continue
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            area = bbox[2] * bbox[3]
            if area > max_area and bbox[2] > 0 and bbox[3] > 0:
                max_area = area
                best_detection = detection
        except Exception as e:
             # print(f"Lỗi nhỏ khi xử lý detection: {e}")
             continue

    if best_detection is None:
        # print("Không thể xác định khuôn mặt chính.")
        return None

    # Lấy bounding box của khuôn mặt chính
    bboxC = best_detection.location_data.relative_bounding_box
    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                 int(bboxC.width * iw), int(bboxC.height * ih)

    # Cắt ảnh khuôn mặt (không cần padding nhiều vì resize sẽ xử lý)
    x, y = max(0, x), max(0, y) # Đảm bảo không âm
    face_image = image[y : y + h, x : x + w]

    if face_image.size == 0:
        # print("Ảnh khuôn mặt cắt ra bị rỗng.")
        return None

    # Tiền xử lý và tạo embedding
    embedding = preprocess_and_embed(face_image)
    return embedding


def get_face_embedding(image_input):
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
        if image is None:
            print(f"Lỗi: Không thể đọc file ảnh {image_input}.")
            return None
    elif isinstance(image_input, io.BytesIO):
        image_bytes = image_input.getvalue()
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            print("Lỗi: Không thể giải mã dữ liệu bytes thành hình ảnh.")
            return None
    else:
        print("Lỗi: Đầu vào không hợp lệ. Cần là đường dẫn tệp (str) hoặc io.BytesIO.")
        return None

    # Gọi hàm để lấy embedding
    embedding = get_face_embedding_tflite(image)
    return embedding

def get_embeddings_data():
    try:
        if not os.path.isfile(EMBEDDINGS_FILE):
            raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_FILE}")
        with open(EMBEDDINGS_FILE, "rb") as f:
            data = pickle.load(f)
            known_face_embeddings = data["embeddings"]
            known_face_ids = data["ids"]
            if not known_face_embeddings or not known_face_ids:
                print("Cảnh báo: File embeddings trống hoặc không hợp lệ.")
                known_face_embeddings = []
                known_face_ids = []
            print(f"Đã tải {len(known_face_ids)} embeddings TFLite đã biết.")

            return known_face_embeddings, known_face_ids
        
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file embeddings '{EMBEDDINGS_FILE}'.")
        print("Vui lòng chạy lại file 'embedding_face.py' (phiên bản TFLite) trước.")
        exit()
    except Exception as e:
        print(f"Lỗi khi tải file embeddings: {e}")
        exit()


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
        embedding = get_face_embedding_tflite(image)

        if embedding is not None:
            known_face_embeddings.append(embedding)
            known_face_ids.append(username)
            print(f"-> Đã tạo embedding TFLite thành công cho {username}.")
        else:
            print(f"-> Không thể tạo embedding TFLite cho {username} từ ảnh {image_file}.")

    if not known_face_embeddings:
        print("Không tạo được embedding nào. Vui lòng kiểm tra lại ảnh đầu vào và model TFLite.")
        face_detection.close()
        exit()

    # Lưu embeddings và tên vào file pickle (dùng tên file mới)
    data = {"embeddings": known_face_embeddings, "ids": known_face_ids}
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(data, f)

    print(f"\nHoàn tất! Đã lưu {len(known_face_embeddings)} embeddings TFLite vào file: {EMBEDDINGS_FILE}")
    face_detection.close()