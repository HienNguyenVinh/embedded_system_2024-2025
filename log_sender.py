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
    return b64_str


def send_history_log(frame, people_id: str, mode: str = "secure") -> requests.Response:
    timestamp = datetime.utcnow().isoformat() + 'Z'
    image_b64 = _frame_to_base64(frame)

    payload = {
        "timestamp": timestamp,
        "peopleId": people_id,
        "image": image_b64,
        "mode": mode
    }

    url = f"{api_base_url.rstrip('/')}/api/history"
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response


def send_warning_log(frame, info: str) -> requests.Response:
    timestamp = datetime.utcnow().isoformat() + 'Z'
    image_b64 = _frame_to_base64(frame)

    payload = {
        "timestamp": timestamp,
        "image": image_b64,
        "info": info
    }

    url = f"{api_base_url.rstrip('/')}/api/warnings"
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response
