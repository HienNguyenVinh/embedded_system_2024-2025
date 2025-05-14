import asyncio
import websockets
import uuid
import json
import logging
from urllib.parse import urlparse
import time
from settings import SERVER_URI, SUBSCRIBE_TOPIC, RECONNECT_DELAY_SECONDS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Hàm xử lý STOMP ---
async def send_stomp_frame(websocket, command, headers, body=""):
    """Hàm trợ giúp để gửi một khung STOMP."""
    frame = f"{command}\n"
    for key, value in headers.items():
        frame += f"{key}:{value}\n"
    frame += "\n"
    frame += body
    frame += "\x00"
    try:
        await websocket.send(frame)
        # logger.debug(f"Sent STOMP Frame: Command={command}, Headers={headers}")
        if command == "CONNECT" or command == "SUBSCRIBE":
             logger.info(f"Sent STOMP Frame: Command={command}")
    except websockets.exceptions.ConnectionClosed:
        logger.warning(f"Failed to send STOMP frame ({command}): Connection closed.")
    except Exception as e:
        logger.error(f"Error sending STOMP frame ({command}): {e}", exc_info=True)


async def parse_and_process_message(message_body, handlers):
    """
    Phân tích nội dung JSON và gọi handler tương ứng.
    """
    try:
        data = json.loads(message_body)

        # Hỗ trợ cả key 'type'/'payload' và 'message'/'data'
        notification_type = data.get('type') or data.get('message')
        payload = data.get('payload') or data.get('data')

        logger.info(
            f"Received notification: Type='{notification_type}', "
            f"Payload Keys: {list(payload.keys()) if isinstance(payload, dict) else 'None'}"
        )

        if not notification_type or payload is None:
            logger.warning(f"Invalid message format (missing type/message or payload/data): {data}")
            return

        # Chuẩn hóa tên notification để so khớp
        nt = notification_type.upper()

        if nt in ("MODE_UPDATE", "UPDATE_MODE") and handlers.get("update_mode"):
            await handlers["update_mode"](payload.get('mode'))
        elif nt == "ADD_PEOPLE" and handlers.get("add_user"):
            await handlers["add_user"](
                payload.get('identificationId'), payload.get('image_path')
            )
        elif nt == "USER_UPDATED" and handlers.get("update_user"):
            await handlers["update_user"](
                payload.get('id'), payload.get('name'), payload.get('image_path')
            )
        elif nt == "USER_DELETED" and handlers.get("delete_user"):
            await handlers["delete_user"](payload.get('id'))
        else:
            logger.warning(f"No handler or unknown notification type: {notification_type}")

    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from message body: {message_body[:200]}...")
    except Exception:
        logger.exception("Error processing message payload")



async def stomp_websocket_client(shared_data, handlers):
    subscription_id = f"sub-{uuid.uuid4()}"
    last_connect_attempt_time = 0

    while True:
        now = time.time()
        if now - last_connect_attempt_time < RECONNECT_DELAY_SECONDS:
             await asyncio.sleep(RECONNECT_DELAY_SECONDS - (now - last_connect_attempt_time))

        last_connect_attempt_time = time.time()
        logger.info(f"Attempting to connect to STOMP server: {SERVER_URI}")
        websocket = None 

        try:
            parsed_uri = urlparse(SERVER_URI)
            host_value = parsed_uri.hostname
            if parsed_uri.port and \
               ((parsed_uri.scheme == "ws" and parsed_uri.port != 80) or \
                (parsed_uri.scheme == "wss" and parsed_uri.port != 443)):
                host_value = f"{host_value}:{parsed_uri.port}"

            custom_handshake_headers = {"Host": host_value}
            # logger.info(f"Using extra handshake headers: {custom_handshake_headers}")

        except Exception as e:
             logger.error(f"Failed to parse URI or create headers: {e}. Cannot connect.", exc_info=True)
             await asyncio.sleep(RECONNECT_DELAY_SECONDS)
             continue

        try:
            async with websockets.connect(
                SERVER_URI,
                subprotocols=["v10.stomp", "v11.stomp", "v12.stomp"]
            ) as websocket:
                logger.info(f"WebSocket connection established to {SERVER_URI}. Negotiated subprotocol: {websocket.subprotocol}")

                connect_headers = {
                    "accept-version": "1.2,1.1,1.0",
                    "host": "/",
                    "heart-beat": "10000,10000",
                }
                await send_stomp_frame(websocket, "CONNECT", connect_headers)

                stomp_connected = False

                async for message in websocket:
                    lines = message.split('\n')
                    if not lines: continue

                    command = lines[0].strip()

                    if command == 'CONNECTED':
                        stomp_connected = True

                        logger.info(f"STOMP Connection Acknowledged by Server.")
                        # logger.info(f"  Server Version: {server_version}, Session: {session_id}")

                        subscribe_headers = {
                            "id": subscription_id,
                            "destination": SUBSCRIBE_TOPIC,
                            "ack": "auto"
                        }
                        await send_stomp_frame(websocket, "SUBSCRIBE", subscribe_headers)
                        logger.info(f"Subscribed to topic: {SUBSCRIBE_TOPIC} with id: {subscription_id}")

                        # --- Cập nhật trạng thái kết nối chia sẻ ---
                        if "lock" in shared_data and "is_connected" in shared_data:
                            async with shared_data["lock"]:
                                shared_data["is_connected"] = True
                            logger.info("Shared state updated: is_connected = True")

                    elif command == 'MESSAGE':
                        if not stomp_connected:
                             logger.warning("Received MESSAGE frame before STOMP connection was fully established. Ignoring.")
                             continue

                        body_start_index = message.find('\n\n')
                        if body_start_index != -1:
                            body = message[body_start_index + 2:].rstrip('\x00')

                            await parse_and_process_message(body, handlers)
                        else:
                            logger.warning(f"Received MESSAGE frame without body separator: {message[:100]}...")

                    elif command == 'RECEIPT':
                        receipt_id = next((l.split(':', 1)[1].strip() for l in lines[1:] if l.startswith('receipt-id:')), 'N/A')
                        logger.info(f"Received STOMP RECEIPT for id: {receipt_id}")

                    elif command == 'ERROR':
                        stomp_connected = False
                        logger.error(f"Received STOMP ERROR frame:\n{message}")
                        # Cập nhật trạng thái khi có lỗi STOMP
                        if "lock" in shared_data and "is_connected" in shared_data:
                             async with shared_data["lock"]:
                                 shared_data["is_connected"] = False
                             logger.warning("Shared state updated: is_connected = False due to STOMP ERROR")
                        await websocket.close(code=1011, reason="STOMP Protocol Error Received")
                        break 

                    elif command == '':
                        logger.debug("Received Heartbeat Frame (empty line)")

                    else:
                         logger.warning(f"Received unknown STOMP frame type: {command}")

        except websockets.exceptions.ConnectionClosedOK:
            logger.info("WebSocket connection closed normally.")
        except websockets.exceptions.ConnectionClosedError as e:
            logger.error(f"WebSocket connection closed with error: code={e.code}, reason='{e.reason}'")
        except websockets.exceptions.InvalidURI as e:
            logger.error(f"Invalid WebSocket URI: {e}. Check SERVER_URI.")
            await asyncio.sleep(60)
        except websockets.exceptions.InvalidHandshake as e:
             logger.error(f"WebSocket handshake failed: {e}. Check URI, headers, subprotocols, or server logs.")
             await asyncio.sleep(30)
        except websockets.exceptions.WebSocketException as e:
            logger.error(f"WebSocket connection failed: {e}", exc_info=True)
        except ConnectionRefusedError:
            logger.error(f"Connection refused by the server at {SERVER_URI}.")
        except asyncio.TimeoutError:
            logger.error(f"Connection attempt to {SERVER_URI} timed out.")
        except OSError as e:
             logger.error(f"Network OS Error: {e}. Check network connectivity.")
        except Exception as e:
            logger.error(f"An unexpected error occurred in the WebSocket client loop: {e}", exc_info=True)
        finally:
            logger.warning(f"Connection loop ended. Setting shared state is_connected = False.")
            if "lock" in shared_data and "is_connected" in shared_data:
                 async with shared_data["lock"]:
                    shared_data["is_connected"] = False
            if websocket and not websocket.closed:
                try:
                    await websocket.close()
                except Exception as close_err:
                     logger.warning(f"Error during websocket close: {close_err}")

            logger.info(f"Waiting {RECONNECT_DELAY_SECONDS} seconds before reconnecting...")


if __name__ == "__main__":
    async def dummy_handler(msg):
        print(f"Dummy handler received: {msg}")

    test_handlers = {
        "update_mode": dummy_handler,
        "add_user": dummy_handler,
    }
    test_shared_data = {
        "lock": asyncio.Lock(),
        "is_connected": False
    }
    try:
         asyncio.run(stomp_websocket_client(test_shared_data, test_handlers))
    except KeyboardInterrupt:
         print("Test client stopped.")