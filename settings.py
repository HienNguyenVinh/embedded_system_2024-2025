import os

KNOWN_FACES_DIR = os.path.join("data", "known_faces")
EMBEDDINGS_DIR = os.path.join("data", "embeddings")
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "known_faces_embeddings_tflite.pkl")

TFLITE_MODEL_PATH = r"data\model\facenet\facenet_512.tflite"
# TFLITE_MODEL_PATH = "data\model\MobileFaceNet.tflite"
# TFLITE_MODEL_PATH = "data\model\mobile_face_net.tflite"
# TFLITE_MODEL_PATH_3 = r"data\model\facenet\facenet.tflite"
SPOOF_MODEL_DIR = r"data\model\facenet"

DETECTION_CONFIDENCE = 0.7
RECOGNITION_THRESHOLD = 0.35

# Input size của model TFLite
INPUT_WIDTH = 160
INPUT_HEIGHT = 160

NORM_MEAN = 127.5
NORM_STD = 127.5
