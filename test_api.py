import asyncio
import websockets
import logging
import uuid # Để tạo ID subscription duy nhất

# Cấu hình logging cơ bản
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Thông tin kết nối và topic
SERVER_URI = "ws://159.223.90.88:8080/ws/websocket"
SUBSCRIBE_TOPIC = "/topic/messages"
# Tạo một ID subscription duy nhất cho mỗi lần chạy hoặc mỗi subscription
SUBSCRIPTION_ID = f"sub-{uuid.uuid4()}"

async def send_stomp_frame(websocket, command, headers, body=""):
    """Hàm trợ giúp để gửi một khung STOMP."""
    frame = f"{command}\n"
    for key, value in headers.items():
        frame += f"{key}:{value}\n"
    frame += "\n" # Dòng trống ngăn cách header và body
    frame += body
    frame += "\x00" # Ký tự NULL kết thúc khung STOMP
    await websocket.send(frame)
    logging.info(f"Sent STOMP Frame:\n{command}\n{headers}")

async def receive_messages(websocket):
    """Vòng lặp để nhận và xử lý tin nhắn đến."""
    try:
        async for message in websocket:
            # Xử lý tin nhắn nhận được từ server
            # Lưu ý: Tin nhắn nhận được cũng sẽ ở định dạng khung STOMP (MESSAGE, ERROR, RECEIPT, CONNECTED...)
            # Cần phân tích cú pháp khung STOMP để lấy nội dung thực sự nếu cần.
            # Ví dụ đơn giản là chỉ in ra màn hình
            logging.info(f"Received Raw STOMP Frame:\n{message[:500]}...") # Giới hạn độ dài log

            # == Ví dụ phân tích cơ bản (chỉ lấy body của frame MESSAGE) ==
            lines = message.split('\n')
            if len(lines) > 1 and lines[0] == 'MESSAGE':
                body_start_index = message.find('\n\n')
                if body_start_index != -1:
                    # +2 để bỏ qua hai ký tự '\n'
                    # Bỏ qua ký tự NULL ở cuối nếu có
                    body = message[body_start_index + 2:].rstrip('\x00')
                    logging.info(f"Extracted Message Body: {body}")
            elif len(lines) > 1 and lines[0] == 'CONNECTED':
                 logging.info("STOMP Connection Acknowledged by Server.")
            elif len(lines) > 1 and lines[0] == 'ERROR':
                 logging.error(f"Received STOMP ERROR frame:\n{message}")


    except websockets.exceptions.ConnectionClosedOK:
        logging.info("WebSocket connection closed normally.")
    except websockets.exceptions.ConnectionClosedError as e:
        logging.error(f"WebSocket connection closed with error: {e}")
    except Exception as e:
        logging.error(f"An error occurred during message receiving: {e}")

async def connect_and_subscribe():
    """Kết nối tới WebSocket server, gửi lệnh STOMP CONNECT và SUBSCRIBE."""
    try:
        # Kết nối tới WebSocket server
        async with websockets.connect(SERVER_URI) as websocket:
            logging.info(f"Successfully connected to WebSocket server: {SERVER_URI}")

            # 1. Gửi khung STOMP CONNECT
            #    Heart-beat: yêu cầu server và client gửi tín hiệu sống sau mỗi 10s
            connect_headers = {
                "accept-version": "1.2,1.1,1.0", # Các phiên bản STOMP client hỗ trợ
                "heart-beat": "10000,10000"      # client gửi, client nhận (ms)
            }
            await send_stomp_frame(websocket, "CONNECT", connect_headers)

            # Client nên chờ khung CONNECTED từ server trước khi gửi tiếp,
            # nhưng để đơn giản, ta gửi luôn SUBSCRIBE.
            # Trong ứng dụng thực tế, bạn nên có logic chờ và xử lý CONNECTED.

            # 2. Gửi khung STOMP SUBSCRIBE
            subscribe_headers = {
                "id": SUBSCRIPTION_ID,          # ID duy nhất cho subscription này
                "destination": SUBSCRIBE_TOPIC, # Topic muốn đăng ký
                "ack": "auto"                   # Chế độ acknowledge tự động
            }
            await send_stomp_frame(websocket, "SUBSCRIBE", subscribe_headers)
            logging.info(f"Subscribed to topic: {SUBSCRIBE_TOPIC} with id: {SUBSCRIPTION_ID}")

            # 3. Bắt đầu nhận tin nhắn
            logging.info("Waiting for messages...")
            await receive_messages(websocket)

    except websockets.exceptions.InvalidURI as e:
        logging.error(f"Invalid WebSocket URI: {e}")
    except websockets.exceptions.WebSocketException as e:
        logging.error(f"WebSocket connection failed: {e}")
    except ConnectionRefusedError:
        logging.error(f"Connection refused by the server at {SERVER_URI}.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Chạy vòng lặp sự kiện asyncio
    try:
        asyncio.run(connect_and_subscribe())
    except KeyboardInterrupt:
        logging.info("Client stopped manually.")