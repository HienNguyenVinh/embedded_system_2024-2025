from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import asyncio
import websockets
import json
import base64 
import io          
import pickle
from settings import *
from PIL import Image
from settings import *
from embedding_face import get_face_embedding_tflite, get_embeddings_data
from face_camera import main_camera_loop, init_face_spoof_detector, init_face_recognition_model
from face_detector import FaceDetector
from picamera2 import Picamera2
import cv2

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')

# ----- Cấu hình -----
BACKEND_WS_URL = "ws://localhost:8080/smartdoor-pi-websocket/websocket"
RECONNECT_DELAY_SECONDS = 5
CAMERA_INDEX = 0 

# shared_state_lock = threading.Lock()
shared_state_lock = None
current_mode = "secure" 
is_connected_to_backend = False

known_face_embeddings, known_face_ids = get_embeddings_data()
known_faces_data_cache = {}
# --- Hàm xử lý WebSocket (Chạy trong thread riêng) ---

def update_local_mode(new_mode):
    """Cập nhật chế độ hoạt động (thread-safe)."""
    global current_mode
    with shared_state_lock: # Khóa trước khi thay đổi
        if current_mode != new_mode:
            current_mode = new_mode
            # print(f"Mode updated locally to: {current_mode}")

def process_user_image_data(user_id, user_name, base64_image_data):
    """Giải mã ảnh, tạo encoding và cập nhật (thread-safe)."""
    global known_face_embeddings, known_face_ids, known_faces_data_cache
    if not base64_image_data:
        # printing(f"No image data provided for user {user_id} ({user_name}). Skipping encoding.")
        # Cập nhật cache thông tin không có ảnh
        with shared_state_lock:
            known_faces_data_cache[user_id] = {"name": user_name, "has_encoding": False}
            # Xóa encoding cũ nếu có
            if user_id in known_face_ids:
                indices_to_remove = [i for i, id_val in enumerate(known_face_ids) if id_val == user_id]
                for index in sorted(indices_to_remove, reverse=True):
                    del known_face_embeddings[index]
                    del known_face_ids[index]
                # print(f"Removed previous encoding for user {user_id}.")
        return

    try:
        # Giải mã Base64 -> bytes
        image_bytes = base64.b64decode(base64_image_data)
        # print(f"Decoded image for user {user_id}, size: {len(image_bytes)} bytes.")

        # Tạo file object trong bộ nhớ từ bytes
        image_file = io.BytesIO(image_bytes)
        img = Image.open(image_file)

        embedding = get_face_embedding(img)

        # Cập nhật danh sách encodings và IDs (thread-safe)
        with shared_state_lock:
            # Xóa encoding cũ nếu user đã tồn tại
            if user_id in known_face_ids:
                indices_to_remove = [i for i, id_val in enumerate(known_face_ids) if id_val == user_id]
                for index in sorted(indices_to_remove, reverse=True):
                    del known_face_embeddings[index]
                    del known_face_ids[index]
                # print(f"Removed previous encoding to update user {user_id}.")

            # Thêm encoding mới
            known_face_embeddings.append(embedding)
            known_face_ids.append(user_id)
            known_faces_data_cache[user_id] = {"name": user_name, "has_encoding": True} # Cập nhật cache
            # print(f"User {user_id} ({user_name}) added/updated with new encoding.")
            # print(f"Total known encodings: {len(known_face_embeddings)}")

    except base64.binascii.Error:
        print(f"Invalid Base64 data received for user {user_id}.")
    except Exception as e:
        print(f"Error processing image for user {user_id}: {e}", exc_info=True)

def remove_local_user(user_id):
    """Xóa thông tin và encoding người dùng (thread-safe)."""
    global known_face_embeddings, known_face_ids, known_faces_data_cache
    with shared_state_lock: # Khóa toàn bộ thao tác xóa
        if user_id in known_face_ids:
            removed_user_name = known_faces_data_cache.get(user_id, {}).get('name', 'N/A')
            indices_to_remove = [i for i, id_val in enumerate(known_face_ids) if id_val == user_id]
            for index in sorted(indices_to_remove, reverse=True):
                del known_face_embeddings[index]
                del known_face_ids[index]
            if user_id in known_faces_data_cache:
                del known_faces_data_cache[user_id] # Xóa khỏi cache
            # print(f"User {user_id} ({removed_user_name}) and their encoding removed locally.")
            # print(f"Total known encodings: {len(known_face_embeddings)}")
        elif user_id in known_faces_data_cache:
             del known_faces_data_cache[user_id] # Xóa khỏi cache nếu chỉ có thông tin mà ko có encoding
            #  print(f"User {user_id} (no encoding found) removed from cache.")
        else:
            print(f"Received delete request for unknown user ID: {user_id}")

async def websocket_listener_task():
    """Task chạy trong thread riêng để lắng nghe WebSocket."""
    global is_connected_to_backend
    while True:
        # print(f"Attempting WebSocket connection to {BACKEND_WS_URL}")
        try:
            async with websockets.connect(BACKEND_WS_URL) as websocket:
                # print("WebSocket connection established.")
                with shared_state_lock: # Cập nhật trạng thái kết nối
                    is_connected_to_backend = True

                # --- Có thể thêm logic lấy trạng thái ban đầu ở đây ---
                # Ví dụ: gửi một message yêu cầu trạng thái ban đầu sau khi kết nối
                # await websocket.send(json.dumps({"action": "get_initial_state"}))
                print("Requested initial state from backend.")
                # ----------------------------------------------------

                async for message in websocket:
                    try:
                        # Parse message (tương tự như trước)
                        json_start_index = message.find('{')
                        if json_start_index != -1:
                            json_payload = message[json_start_index:]
                            data = json.loads(json_payload)
                            # print(f"Received notification: {data.get('type')}")
                            print(f"Full payload: {data}") # Gỡ lỗi

                            notification_type = data.get('type')
                            payload = data.get('payload')

                            if not notification_type or payload is None:
                                #  printing(f"Invalid message format: {data}")
                                 continue

                            # Xử lý (các hàm đã được làm thread-safe)
                            if notification_type == "MODE_UPDATE":
                                update_local_mode(payload.get('mode'))
                            elif notification_type == "USER_ADDED":
                                process_user_image_data(payload.get('id'), payload.get('name'), payload.get('faceImageData'))
                            elif notification_type == "USER_UPDATED":
                                process_user_image_data(payload.get('id'), payload.get('name'), payload.get('faceImageData'))
                            elif notification_type == "USER_DELETED":
                                remove_local_user(payload.get('id'))
                            else:
                                print(f"Unknown notification type: {notification_type}")
                        else:
                             print(f"Received non-JSON or unexpected format: {message[:100]}...")

                    except json.JSONDecodeError:
                        print(f"Failed JSON decode: {message[:100]}...")
                    except Exception as e:
                        print(f"Error processing message: {e}", exc_info=True)

        except Exception as e: # Bắt mọi lỗi kết nối/runtime trong websocket client
            print(f"WebSocket error: {e}. Reconnecting in {RECONNECT_DELAY_SECONDS}s...")
        finally:
             # Đặt trạng thái là chưa kết nối khi vòng lặp kết nối kết thúc (do lỗi hoặc đóng)
             with shared_state_lock:
                  is_connected_to_backend = False
            #  print("WebSocket connection lost or closed.")
             await asyncio.sleep(RECONNECT_DELAY_SECONDS) # Đợi trước khi thử lại

def run_websocket_listener():
    """Hàm để chạy asyncio event loop trong thread mới."""
    asyncio.run(websocket_listener_task())


# app = FastAPI()
face_detector = FaceDetector()


def generate_frames():
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (640, 480)
    picam2.start()

    while True:
        frame = picam2.capture_array()           # RGB

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        faces = face_detector.process(frame_bgr)     

        if faces is not None:
            ih, iw = frame_bgr.shape[:2]
            for det in faces:
                x, y, w, h, score = det[:5]

                x, y = max(0, x), max(0, y)
                w = min(w, iw - x)
                h = min(h, ih - y)

                cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame_bgr,
                    f"{score/100:.2f}",          
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )

        ret, buf = cv2.imencode(".jpg", frame_bgr)
        if not ret:
            continue
        jpg_bytes = buf.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            jpg_bytes +
            b"\r\n"
        )


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
      <head><title>Stream</title></head>
      <body>
        <h1>Live Face Detection</h1>
        <!-- giữ 640×480 hoặc dùng responsive đúng tỉ lệ -->
        <img src="/video_feed" width="640" height="480" style="object-fit: none;" />
      </body>
    </html>
    """


@app.get("/video_feed")
def video_feed():
    # Streaming endpoint
    return StreamingResponse(
        generate_frames(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == "__main__":
    # uvicorn test_stream:app --host 0.0.0.0 --port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)