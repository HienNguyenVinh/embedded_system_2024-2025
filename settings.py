import os

API_BASE_URL = "http://209.97.160.79/"
SERVER_URI = "ws://209.97.160.79:80/ws/websocket"
SUBSCRIBE_TOPIC = "/topic/messages"
RECONNECT_DELAY_SECONDS = 5
CAMERA_INDEX = 0 

KNOWN_FACES_DIR = os.path.join("data", "known_faces")
EMBEDDINGS_DIR = os.path.join("data", "embeddings")
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "known_faces_embeddings_tflite.pkl")

TFLITE_MODEL_PATH = "/home/hien/embedded_system_2024-2025/data/model/facenet/facenet_512.tflite"
SPOOF_MODEL_PATH = "/home/hien/embedded_system_2024-2025/data/model/facenet"
DETECTION_MODEL_PATH = 'data/model/facenet/face_detection_yunet_2023mar.onnx'

HAAR_CASCADE_PATH = '/home/hien/embedded_system_2024-2025/data/model/haarcascade_frontalface_default.xml'

DETECTION_CONFIDENCE = 0.7
RECOGNITION_THRESHOLD = 0.35

# Input size của model face embedding
INPUT_WIDTH = 160
INPUT_HEIGHT = 160

NORM_MEAN = 127.5
NORM_STD = 127.5
