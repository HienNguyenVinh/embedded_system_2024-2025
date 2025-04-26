import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import tensorflow as tf
from scipy.spatial.distance import cosine
from settings import *
from face_spoof_detector import FaceSpoofDetector
from PIL import Image
import time

def init_face_recognition_model():
    try:
        # Ensure the path comes from settings.py or is defined correctly
        if not os.path.isfile(TFLITE_MODEL_PATH):
            raise FileNotFoundError(f"Recognition TFLite model not found: {TFLITE_MODEL_PATH}")
        interpreter_rec = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
        interpreter_rec.allocate_tensors()
        # input_details_rec = interpreter_rec.get_input_details()
        
        print("TFLite model loaded for real-time recognition.")
        return interpreter_rec
    
    except Exception as e:
        print(f"Lỗi khi load TFLite model nhận dạng '{TFLITE_MODEL_PATH}': {e}")
        exit()

def init_face_spoof_detector():
    try:
        spoof_detector = FaceSpoofDetector(model_dir=SPOOF_MODEL_DIR, use_gpu=False)
        return spoof_detector
    
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy thư mục hoặc tệp mô hình Spoof: {e}")
        print("Vui lòng đảm bảo SPOOF_MODEL_DIR trong settings.py là chính xác và chứa các tệp .tflite cần thiết.")
        exit()
    except Exception as e:
        print(f"Lỗi khi khởi tạo FaceSpoofDetector: {e}")
        exit()


def preprocess_and_embed_realtime(face_image, interpreter_rec, input_details_rec, output_details_rec, input_dtype_rec):
    """Tiền xử lý và tạo embedding nhận dạng cho frame thời gian thực."""
    try:
        # Resize for recognition model
        resized_face = cv2.resize(face_image, (INPUT_WIDTH, INPUT_HEIGHT))
        # Convert BGR (from OpenCV) to RGB
        rgb_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
        # Normalize (Assuming NORM_MEAN and NORM_STD are defined in settings.py)
        normalized_face = (rgb_face.astype(np.float32) - NORM_MEAN) / NORM_STD
        input_data = np.expand_dims(normalized_face, axis=0)

        # Ensure data type matches model input
        if input_data.dtype != input_dtype_rec:
            input_data = input_data.astype(input_dtype_rec)

        # Run inference for recognition
        interpreter_rec.set_tensor(input_details_rec[0]['index'], input_data)
        interpreter_rec.invoke()
        embedding = interpreter_rec.get_tensor(output_details_rec[0]['index'])

        return embedding[0]
    except Exception as e:
        # print(f"Error in preprocess/embed realtime: {e}")
        return None

def find_matching_face(face_embedding, known_face_embeddings, known_face_ids):
    """So sánh embedding nhận dạng với CSDL đã biết."""
    if face_embedding is None or not known_face_embeddings:
        return "Unknown", float('inf') # Return infinity distance if no known faces or error

    try:
        distances = [cosine(face_embedding, known_embedding) for known_embedding in known_face_embeddings]
    except Exception as e:
        print(f"Error calculating distances: {e}")
        return "Error", float('inf')

    if not distances: return "Unknown", float('inf') # Should not happen if known_face_embeddings is checked, but good practice

    min_distance_index = np.argmin(distances)
    min_distance = distances[min_distance_index]

    # print(f"Min distance: {min_distance:.4f}") # Debugging

    if min_distance < RECOGNITION_THRESHOLD:
        return known_face_ids[min_distance_index], min_distance
    else:
        return "Unknown", min_distance

def main_camera_loop(current_mode, is_connected_to_backend, known_face_embeddings, known_face_ids, shared_state_lock, spoof_detector, face_detector, face_recognitor):
    print("Starting main camera processing loop...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Lỗi: Không thể mở camera.")
        exit()

    last_connection_check_time = time.time()
    effective_mode = "secure"

    input_details_rec = face_recognitor.get_input_details()
    output_details_rec = face_recognitor.get_output_details()
    input_dtype_rec = input_details_rec[0]['dtype']

    while True:
        current_time = time.time()
        local_is_connected = False
        local_current_mode_from_ws = "secure"
        local_embeddings = []
        local_ids = []
        # with shared_state_lock:
        #      local_is_connected = is_connected_to_backend
        #      local_current_mode_from_ws = current_mode
        #      # Sao chép dữ liệu để tránh giữ lock quá lâu khi xử lý ảnh
        #      local_embeddings = list(known_face_embeddings)
        #      local_ids = list(known_face_ids)

        local_embeddings = list(known_face_embeddings)
        local_ids = list(known_face_ids)

        if not local_is_connected:
            # Nếu không kết nối, buộc chạy ở chế độ secure
            effective_mode = "secure"
            if current_time - last_connection_check_time > 10: # Log định kỳ nếu mất kết nối
                print("Backend connection lost. Forcing SECURE mode.")
                last_connection_check_time = current_time
        else:
            # Nếu kết nối, sử dụng mode từ WebSocket
            effective_mode = local_current_mode_from_ws
            last_connection_check_time = current_time # Reset thời gian check

        # print(f"Effective Mode: {effective_mode} | Connected: {local_is_connected} | Known Faces: {len(local_embeddings)}")

        success, frame = cap.read()
        if not success:
            print("Lỗi: Không thể đọc frame từ camera hoặc video kết thúc.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- Xử lý logic dựa trên effective_mode ---
        if effective_mode == "secure":
            rgb_frame.flags.writeable = False
            results = face_detector.process(rgb_frame) 
            pil_frame_rgb = Image.fromarray(rgb_frame)
            frame.flags.writeable = True 

            if results.detections:
                ih, iw, _ = frame.shape
                for detection in results.detections:
                    try:
                        # Extract bounding box (relative -> absolute)
                        bboxC = detection.location_data.relative_bounding_box
                        if bboxC is None: continue
                        # Ensure box coordinates are within frame boundaries and valid
                        x = max(0, int(bboxC.xmin * iw))
                        y = max(0, int(bboxC.ymin * ih))
                        w = min(iw - x, int(bboxC.width * iw))
                        h = min(ih - y, int(bboxC.height * ih))

                        # Skip invalid or zero-sized boxes
                        if w <= 0 or h <= 0: continue

                        # Crop face for recognition model input
                        face_image_rec = frame[y : y + h, x : x + w]
                        if face_image_rec.size == 0: continue

                        # Get recognition embedding
                        current_embedding = preprocess_and_embed_realtime(face_image_rec, face_recognitor, input_details_rec, output_details_rec, input_dtype_rec)

                        # Find best match
                        recognition_name, recognition_distance = find_matching_face(current_embedding, local_embeddings, local_ids)

                        bbox_spoof = (x, y, w, h)
                        spoof_results = spoof_detector.detect_spoof(pil_frame_rgb, bbox_spoof)
                        is_spoof = spoof_results['is_spoof']
                        spoof_score = spoof_results['score']
                        
                        if recognition_name != "Unknown" and not is_spoof:
                            color = (0, 255, 0)
                            status_text = "REAL"
                            # print(f"Recognition successful: Name: {recognition_name}, Distance: {recognition_distance}")
                            # --- MỞ CỬA ---
                            # open_door()
                            # --- GỬI LOG HISTORY (POST /api/history) ---
                            # send_history_log(recognized_id, frame, effective_mode) # Cần hàm này

                        elif is_spoof:
                            color = (0, 0, 255)
                            status_text = f"SPOOF ({spoof_score:.2f})"
                            # print("Face spoof detected!")
                        else:
                            color = (255, 0, 0) 
                            status_text = "REAL"
                            # print(f"Unknown face detected: {recognition_name}, Distance: {recognition_distance}")
                            # --- KIỂM TRA LASER VÀ KÍCH HOẠT CẢNH BÁO ---
                            # if is_laser_triggered():
                            #     activate_buzzer()
                            #     send_warning_log("Laser triggered, unknown face", frame) # Cần hàm này

                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                        rec_text = f"{recognition_name}"
                        cv2.putText(frame, rec_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        cv2.putText(frame, status_text, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    except Exception as e:
                        continue

            else:
                 # Không có khuôn mặt nào được phát hiện
                 # --- KIỂM TRA LASER VÀ KÍCH HOẠT CẢNH BÁO ---
                 # if is_laser_triggered():
                 #     activate_buzzer()
                 #     send_warning_log("Laser triggered, no face detected", frame) # Cần hàm này
                 pass 
            
        elif effective_mode == "free":
            # Chế độ tự do, ví dụ: chỉ log khi laser bị chặn
            # if is_laser_triggered():
            #    print("Laser triggered in FREE mode. Logging history.")
            #    # Cố gắng nhận diện nhanh nếu có thể
            #    recognized_id = try_recognize_face_quickly(frame, local_embeddings, local_ids) # Cần hàm này
            #    send_history_log(recognized_id, frame, effective_mode)
            pass

        cv2.imshow("Face Recognition & Spoof Detection (Nhan 'q' de thoat)", frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Thoát bởi người dùng.")
            break

    # Giải phóng camera khi kết thúc
    cap.release()
    cv2.destroyAllWindows()
    print("Main camera loop stopped.")