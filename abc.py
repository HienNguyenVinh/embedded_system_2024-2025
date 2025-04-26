import cv2

def list_cameras(max_index=10):
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            available.append(i)
        cap.release()
    return available

cameras = list_cameras()
print("Camera được kết nối:", cameras)
