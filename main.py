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
from picamera2 import Picamera2
import cv2

from embedding_face import get_face_embedding_tflite, get_embeddings_data
from camera_loop import main_camera_loop
from face_utils import init_face_recognition_model, init_face_spoof_detector
from face_detector import FaceDetector
from door_control import DoorController
from log_sender import send_history_log, send_warning_log

shared_state_lock = asyncio.Lock()
shared_data = {
    "lock": shared_state_lock,
    "is_connected_to_backend": False,
    "current_mode": "FREE",
    "known_face_embeddings": [],
    "known_face_ids": [],
}

known_face_embeddings, known_face_ids = get_embeddings_data()
known_faces_data_cache = {}

def load_embeddings():
    global shared_data
    try:
        asyncio.run(update_shared_data(known_face_embeddings, known_face_ids))
        logger.info(f"Loaded {len(shared_data['known_face_ids'])} known faces from {EMBEDDINGS_FILE}")
    except Exception as e:
        logger.error(f"Error loading embeddings: {e}", exc_info=True)

def save_embeddings():
    global shared_data
    try:
        data_to_save = {
            "embeddings": shared_data['known_face_embeddings'],
            "ids": shared_data['known_face_ids']
        }
        with open(EMBEDDINGS_FILE, "wb") as f:
            pickle.dump(data_to_save, f)
        logger.info(f"Saved {len(shared_data['known_face_ids'])} faces to {EMBEDDINGS_FILE}")
    except Exception as e:
        logger.error(f"Error saving embeddings: {e}", exc_info=True)

async def update_shared_data(**kwargs):
    """Cập nhật shared_data một cách an toàn."""
    async with shared_data["lock"]:
        for key, value in kwargs.items():
            if key in shared_data:
                 # logger.debug(f"Updating shared_data: {key} = {value}") # Log nếu cần
                 shared_data[key] = value
            elif key == "embeddings": # Xử lý riêng cho embeddings và ids nếu cần
                 shared_data["known_face_embeddings"] = value
            elif key == "ids":
                 shared_data["known_face_ids"] = value
            else:
                 logger.warning(f"Attempted to update unknown key in shared_data: {key}")


async def update_local_mode(new_mode):
    """Callback khi nhận MODE_UPDATE."""
    if new_mode:
        logger.info(f"Received MODE_UPDATE: Setting mode to {new_mode}")
        await update_shared_data(current_mode=new_mode)
    else:
        logger.warning("Received MODE_UPDATE with invalid mode.")

async def process_user_image_data(user_id, name, face_image_data_base64):
    """Callback khi nhận USER_ADDED hoặc USER_UPDATED."""
    if not all([user_id, name, face_image_data_base64]):
        logger.warning(f"Received ADD/UPDATE user with missing data (id={user_id}, name={name})")
        return

    logger.info(f"Processing user data for ID: {user_id}, Name: {name}")
    try:
        image_data = base64.b64decode(face_image_data_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logger.error(f"Failed to decode image for user {user_id}")
            return

        embedding = get_face_embedding_tflite(img, face_detector, face_recognitor)

        if embedding is not None and embedding.size > 0:
            logger.info(f"Successfully extracted embedding for user {user_id}.")

            async with shared_data["lock"]:
                shared_data["known_face_embeddings"][user_id] = embedding
                if user_id not in shared_data["known_face_ids"]:
                    shared_data["known_face_ids"].append(user_id)
                    logger.info(f"Added new user ID {user_id} to known list.")
                else:
                     logger.info(f"Updated embedding for existing user ID {user_id}.")

        else:
            logger.warning(f"Could not extract embedding for user {user_id}. Image might not contain a face or is invalid.")

    except base64.binascii.Error:
        logger.error(f"Invalid Base64 data received for user {user_id}")
    except Exception as e:
        logger.error(f"Error processing image data for user {user_id}: {e}", exc_info=True)

async def remove_local_user(user_id):
    """Callback khi nhận USER_DELETED."""
    if not user_id:
         logger.warning("Received USER_DELETED with missing user ID.")
         return

    logger.info(f"Received USER_DELETED: Removing user ID {user_id}")
    async with shared_data["lock"]:
        if user_id in shared_data["known_face_ids"]:
            indices_to_remove = [i for i, id_val in enumerate(shared_data["known_face_ids"]) if id_val == user_id]
            for index in sorted(indices_to_remove, reverse=True):
                shared_data["known_face_embeddings"].remove(user_id)
                shared_data["known_face_ids"].remove(user_id)
            logger.info(f"Removed user ID {user_id} from local data.")
        else:
            logger.warning(f"Attempted to remove non-existent user ID: {user_id}")

app = FastAPI()

try:
    face_detector = FaceDetector()
    spoof_detector = init_face_spoof_detector()
    face_recognitor = init_face_recognition_model()
    door_controller = DoorController()
    logger.info("Face processing models and door controller initialized.")
except Exception as e:
    logger.critical(f"Failed to initialize models or hardware: {e}", exc_info=True)
    exit(1)
    raise RuntimeError("Initialization failed") from e

websocket_task = None

@app.on_event("startup")
async def startup_event():
    global websocket_task
    logger.info("Application startup...")
    load_embeddings()

    logger.info("Starting STOMP WebSocket client task...")
    client_coro = stomp_websocket_client(shared_data, websocket_handlers)
    websocket_task = asyncio.create_task(client_coro)
    logger.info("STOMP WebSocket client task created.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutdown...")
    if websocket_task and not websocket_task.done():
        logger.info("Cancelling WebSocket client task...")
        websocket_task.cancel()
        try:
            await websocket_task
        except asyncio.CancelledError:
            logger.info("WebSocket client task cancelled successfully.")
        except Exception as e:
             logger.error(f"Error during WebSocket task cancellation: {e}", exc_info=True)

    save_embeddings()
    logger.info("Application finished.")

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
        main_camera_loop(shared_data, spoof_detector, face_detector, interpreter_rec, door_controller)
        media_type='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == "__main__":
    # uvicorn test_stream:app --host 0.0.0.0 --port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)