import asyncio
import websockets
import uuid


SERVER_URI = "ws://209.97.160.79:80/ws/websocket"
SUBSCRIBE_TOPIC = "/topic/messages"
SUBSCRIPTION_ID = f"sub-{uuid.uuid4()}"

async def send_stomp_frame(websocket, command, headers, body=""):
    """Hàm trợ giúp để gửi một khung STOMP."""
    frame = f"{command}\n"
    for key, value in headers.items():
        frame += f"{key}:{value}\n"
    frame += "\n" 
    frame += body
    frame += "\x00"
    await websocket.send(frame)
    print(f"Sent STOMP Frame:\n{command}\n{headers}")

async def receive_messages(websocket):
    """Vòng lặp để nhận và xử lý tin nhắn đến."""
    try:
        async for message in websocket:
            print(f"Received Raw STOMP Frame:\n{message[:500]}...") # Giới hạn độ dài log

            # == Ví dụ phân tích cơ bản (chỉ lấy body của frame MESSAGE) ==
            lines = message.split('\n')
            if len(lines) > 1 and lines[0] == 'MESSAGE':
                body_start_index = message.find('\n\n')
                if body_start_index != -1:
                    body = message[body_start_index + 2:].rstrip('\x00')
                    print(f"Extracted Message Body: {body}")
            elif len(lines) > 1 and lines[0] == 'CONNECTED':
                print("STOMP Connection Acknowledged by Server.")
            elif len(lines) > 1 and lines[0] == 'ERROR':
                print(f"Received STOMP ERROR frame:\n{message}")


    except websockets.exceptions.ConnectionClosedOK:
        print("WebSocket connection closed normally.")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"WebSocket connection closed with error: {e}")
    except Exception as e:
        print(f"An error occurred during message receiving: {e}")

async def connect_and_subscribe():
    """Kết nối tới WebSocket server, gửi lệnh STOMP CONNECT và SUBSCRIBE."""
    try:
        # Kết nối tới WebSocket server
        async with websockets.connect(
            SERVER_URI,
            subprotocols=["v10.stomp", "v11.stomp", "v12.stomp"]
        ) as websocket:
            print(f"Successfully connected to WebSocket server: {SERVER_URI}")

            # 1. Gửi khung STOMP CONNECT
            #    Heart-beat: yêu cầu server và client gửi tín hiệu sống sau mỗi 10s
            connect_headers = {
                "accept-version": "1.2,1.1,1.0", 
                "host": "/",   
                "heart-beat": "10000,10000"     
            }
            await send_stomp_frame(websocket, "CONNECT", connect_headers)

            # 2. Gửi khung STOMP SUBSCRIBE
            subscribe_headers = {
                "id": SUBSCRIPTION_ID,          # ID duy nhất cho subscription này
                "destination": SUBSCRIBE_TOPIC, # Topic muốn đăng ký
                "ack": "auto"                   # Chế độ acknowledge tự động
            }
            await send_stomp_frame(websocket, "SUBSCRIBE", subscribe_headers)
            print(f"Subscribed to topic: {SUBSCRIBE_TOPIC} with id: {SUBSCRIPTION_ID}")

            # 3. Bắt đầu nhận tin nhắn
            print("Waiting for messages...")
            await receive_messages(websocket)

    except websockets.exceptions.InvalidURI as e:
        print(f"Invalid WebSocket URI: {e}")
    except websockets.exceptions.WebSocketException as e:
        print(f"WebSocket connection failed: {e}")
    except ConnectionRefusedError:
        print(f"Connection refused by the server at {SERVER_URI}.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Chạy vòng lặp sự kiện asyncio
    try:
        asyncio.run(connect_and_subscribe())
    except KeyboardInterrupt:
        print("Client stopped manually.")