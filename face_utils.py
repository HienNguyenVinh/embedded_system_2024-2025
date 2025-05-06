import os
import tflite_runtime.interpreter as tf
from settings import *
from face_spoof_detector import FaceSpoofDetector

def init_face_recognition_model():
    try:
        # Ensure the path comes from settings.py or is defined correctly
        if not os.path.isfile(TFLITE_MODEL_PATH):
            raise FileNotFoundError(f"Recognition TFLite model not found: {TFLITE_MODEL_PATH}")
        interpreter_rec = tf.Interpreter(model_path=TFLITE_MODEL_PATH)
        interpreter_rec.allocate_tensors()
        # input_details_rec = interpreter_rec.get_input_details()
        
        print("TFLite model loaded for real-time recognition.")
        return interpreter_rec
    
    except Exception as e:
        print(f"Lỗi khi load TFLite model nhận dạng '{TFLITE_MODEL_PATH}': {e}")
        exit()

def init_face_spoof_detector():
    try:
        spoof_detector = FaceSpoofDetector(model_dir=SPOOF_MODEL_PATH, use_gpu=False)
        return spoof_detector
    
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy thư mục hoặc tệp mô hình Spoof: {e}")
        print("Vui lòng đảm bảo SPOOF_MODEL_PATH trong settings.py là chính xác và chứa các tệp .tflite cần thiết.")
        exit()
    except Exception as e:
        print(f"Lỗi khi khởi tạo FaceSpoofDetector: {e}")
        exit()