import base64
import cv2
import requests
from datetime import datetime
from settings import API_BASE_URL


def _frame_to_base64(frame) -> str:
    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
        raise ValueError("Failed to encode frame to JPEG")

    jpg_bytes = buffer.tobytes()
    b64_str = base64.b64encode(jpg_bytes).decode('utf-8')
    # print(b64_str)
    # print(base64.urlsafe_b64encode(jpg_bytes))
    return b64_str

# _frame_to_base64(cv2.imread("/home/hien/embedded_system_2024-2025/data/known_faces/HienNV01.jpg"))



def send_history_log(frame, people_id: int, mode: str = "secure") -> requests.Response:
    timestamp = datetime.utcnow().isoformat() + 'Z'
    image_b64 = _frame_to_base64(frame)

    payload = {
        "timestamp": timestamp,
        "peopleId": people_id,
        "imagePath": image_b64,
        "mode": mode
    }

    # headers = {
    #     "Content-Type": "application/json",
    # }

    url = f"{API_BASE_URL.rstrip('/')}/api/history"
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response


# send_history_log(cv2.imread("/home/hien/embedded_system_2024-2025/data/known_faces/HienNV01.jpg"), "1", "secure")

def send_warning_log(frame, info: str) -> requests.Response:
    timestamp = datetime.utcnow().isoformat() + 'Z'
    image_b64 = _frame_to_base64(frame)

    payload = {
        "timestamp": timestamp,
        "imagePath": image_b64,
        "info": info
    }

    url = f"{API_BASE_URL.rstrip('/')}/api/warning"
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response

# send_warning_log(cv2.imread("/home/hien/embedded_system_2024-2025/data/known_faces/HienNV01.jpg"), "test warning")
