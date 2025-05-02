import cv2
from settings import HAAR_CASCADE_PATH 

class FaceDetector:
    def __init__(self):
        # Load the Haar Cascade classifier
        self.face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        if self.face_cascade.empty():
            print("Lỗi: Không thể tải Haar Cascade classifier!")
        else:
            print("Haar Cascade classifier đã được tải thành công.")

    def process(self, frame):
        """
        Detect faces and return both bounding boxes and confidence scores.

        Returns:
            detections: list of tuples (x, y, w, h, score)
        """
        # Convert to grayscale as Haar works on gray images
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use detectMultiScale2 to get weights (scores) for each detection
        # Note: in some OpenCV versions it's detectMultiScale2, else use detectMultiScale with outputRejectLevels
        try:
            rects, weights = self.face_cascade.detectMultiScale2(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
        except AttributeError:
            # Fallback if detectMultiScale2 is unavailable
            rects = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                outputRejectLevels=False
            )
            # No weights available: assign dummy score 1.0
            weights = [1.0] * len(rects)

        detections = []
        for (rect, score) in zip(rects, weights):
            x, y, w, h = rect
            detections.append((x, y, w, h, float(score * 100)))

        return detections

    def get_faces(self, frame):
        """
        Return cropped face images and metadata including score.

        Returns:
            out: list of tuples (crop, (x, y, w, h, score))
        """
        detections = self.process(frame)
        out = []
        h_frame, w_frame = frame.shape[:2]
        for (x, y, w, h, score) in detections:
            # ensure bounding box inside frame
            x, y = max(0, x), max(0, y)
            w, h = min(w, w_frame - x), min(h, h_frame - y)
            if w > 0 and h > 0:
                crop = frame[y:y+h, x:x+w]
                out.append((crop, (x, y, w, h, score)))
        return out
