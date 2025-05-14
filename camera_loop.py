import cv2
import numpy as np
import os
import tflite_runtime.interpreter as tf
from scipy.spatial.distance import cosine
from PIL import Image
import time
import asyncio
from picamera2 import Picamera2

from settings import *
from face_spoof_detector import FaceSpoofDetector
from embedding_face import get_face_embedding_tflite, preprocess_and_embed
from log_sender import send_history_log, send_warning_log


def find_matching_face(face_embedding, known_face_embeddings, known_face_ids):
    if face_embedding is None or len(known_face_embeddings) == 0:
        return "Unknown", float('inf')

    try:
        distances = [cosine(face_embedding, known_embedding) for known_embedding in known_face_embeddings]
    except Exception as e:
        print(f"Error calculating distances: {e}")
        return "Error", float('inf')

    if not distances: return "Unknown", float('inf')

    min_distance_index = np.argmin(distances)
    min_distance = distances[min_distance_index]

    # print(f"Min distance: {min_distance:.4f}") # Debugging

    if min_distance < RECOGNITION_THRESHOLD:
        return known_face_ids[min_distance_index], min_distance
    else:
        return "Unknown", min_distance

async def main_camera_loop(shared_data, spoof_detector, face_detector, face_recognitor, door_controller):
    print("Starting main camera processing loop...")
    picam = Picamera2()
    picam.preview_configuration.main.size = (640, 480)
    picam.start()

    last_connection_check_time = time.time()
    effective_mode = "secure"

    input_details_rec = face_recognitor.get_input_details()
    output_details_rec = face_recognitor.get_output_details()
    input_dtype_rec = input_details_rec[0]['dtype']

    seen_history = set()  
    seen_warnings = set()
    spoof_warning = False

    while True:
        current_time = time.time()
        local_is_connected = False
        local_current_mode_from_ws = "secure"
        local_embeddings = []
        local_ids = []
        async with shared_data["lock"]:
            local_is_connected = shared_data["is_connected"]
            local_current_mode_from_ws = shared_data["current_mode"]
            local_embeddings = shared_data["known_face_embeddings"].copy()
            local_ids = shared_data["known_face_ids"][:]

        # local_embeddings = list(known_face_embeddings)
        # local_ids = list(known_face_ids)

        if not local_is_connected:
            effective_mode = "secure"
            if current_time - last_connection_check_time > 10:
                print("Backend connection lost. Forcing SECURE mode.")
                last_connection_check_time = current_time
        else:
            effective_mode = local_current_mode_from_ws
            last_connection_check_time = current_time

        # print(f"Effective Mode: {effective_mode} | Connected: {local_is_connected} | Known Faces: {len(local_embeddings)}")

        frame = picam.capture_array()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ret, buf = cv2.imencode(".jpg", rgb_frame)

        if effective_mode == "secure":
            rgb_frame.flags.writeable = False
            face_results = face_detector.get_faces(rgb_frame)
            rgb_frame.flags.writeable = True 

            # print(f"Detected {len(face_results)} faces.")
            if len(face_results) > 0:
                for face_image_rec, (x,y,w,h, score) in face_results:
                    try:
                        current_embedding = preprocess_and_embed(
                            face_image_rec,
                            face_recognitor,
                            input_details_rec,
                            output_details_rec,
                            input_dtype_rec
                        )

                        recognition_id, recognition_distance = find_matching_face(
                            current_embedding,
                            local_embeddings,
                            local_ids
                        )


                        # x, y, w, h = cv2.boundingRect(face_image_rec)
                        pil_frame_rgb = Image.fromarray(rgb_frame)
                        spoof_results = spoof_detector.detect_spoof(pil_frame_rgb, (x, y, w, h))
                        is_spoof = spoof_results["is_spoof"]
                        spoof_score = spoof_results["score"]

                        if recognition_id != "Unknown" and not is_spoof:
                            color = (0, 255, 0)
                            status_text = "REAL"


                            if recognition_id not in seen_history:
                                print(f"Recognition successful: Name: {recognition_id}")
                                send_history_log(frame, recognition_id, effective_mode)
                                seen_history.add(recognition_id)

                                door_controller.activate_buzzer(beeps=2, on_time=0.1, off_time=0.1)
                                door_controller.open_door()

                        elif is_spoof:
                            color = (0, 0, 255)
                            status_text = f"SPOOF ({spoof_score:.2f})"
                            if not spoof_warning:
                                print(f"Face spoof detected!")
                                spoof_warning = True
                                door_controller.activate_buzzer(beeps=3, on_time=0.5, off_time=0.25)
                        else:
                            color = (255, 0, 0) 
                            status_text = "REAL"
                            # print(f"Unknown face detected: {recognition_name}, Distance: {recognition_distance}")

                            # if door_controller.is_ultrasonic_triggered():
                                # door_controller.activate_buzzer()
                                # send_warning_log(frame, "Laser triggered, unknown face")

                            if recognition_id not in seen_history:
                                print(f"Recognition fail: {recognition_id}")
                                door_controller.activate_buzzer(beeps=3, on_time=1, off_time=0.25)
                                send_warning_log(frame, "Unknown face")
                                seen_history.add(recognition_id)

                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, recognition_id, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        cv2.putText(frame, status_text, (x, y + h + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        ret, buf = cv2.imencode(".jpg", frame)
                        
                    except Exception as e:
                        print(f"Error processing face: {e}")
                        continue

            else:
                # Không có khuôn mặt nào được phát hiện
                # if door_controller.is_ultrasonic_triggered():
                    # door_controller.activate_buzzer()
                    # send_warning_log(frame, "Laser triggered, no face detected")
                    # pass
                pass 
            
        elif effective_mode == "free":
            # Chế độ tự do
            # if door_controller.is_ultrasonic_triggered():
            #    print("Laser triggered in FREE mode. Logging history.")
            #    recognized_id = try_recognize_face_quickly(frame, local_embeddings, local_ids)
            #    send_history_log(recognized_id, frame, effective_mode)
            pass

        
        
        if not ret:
            continue
        jpg_bytes = buf.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            jpg_bytes +
            b"\r\n"
        )

        await asyncio.sleep(0.01)

    print("Main camera loop stopped.")