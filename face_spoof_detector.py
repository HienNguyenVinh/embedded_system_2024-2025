import os
import time
import numpy as np
from PIL import Image
import tensorflow as tf
from scipy.spatial.distance import cosine


# --- FaceSpoofDetector Class (Provided) ---
class FaceSpoofDetector:
    """
    Python translation of the Kotlin FaceSpoofDetector using TensorFlow Lite.
    """
    def __init__(
        self,
        model_dir: str,
        use_gpu: bool = False,
        num_threads: int = 4
    ):
        # Scales and model filenames
        self.scales = [2.7, 4.0]
        self.input_dim = 80
        self.output_dim = 3
        self.model_files = [
            "spoof_model_scale_2_7.tflite",
            "spoof_model_scale_4_0.tflite"
        ]

        # Ensure model directory exists
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Spoof model directory not found: {model_dir}")

        # Optionally load GPU delegate
        delegates = []
        if use_gpu:
            try:
                # Adjust the path based on your system if needed
                gpu_delegate = tf.lite.experimental.load_delegate(
                    'libtensorflowlite_gpu_delegate.so'
                )
                delegates.append(gpu_delegate)
                print("GPU delegate loaded for spoof detection.")
            except ValueError:
                print("GPU delegate not supported or found for spoof detection, falling back to CPU.")
            except Exception as e:
                 print(f"Error loading GPU delegate for spoof detection: {e}. Falling back to CPU.")


        # Initialize interpreters
        self.interpreters = []
        for fname in self.model_files:
            path = os.path.join(model_dir, fname)
            if not os.path.isfile(path):
                 raise FileNotFoundError(f"Spoof model file not found: {path}")
            try:
                interpreter = tf.lite.Interpreter(
                    model_path=path,
                    experimental_delegates=delegates,
                    num_threads=num_threads
                )
                interpreter.allocate_tensors()
                self.interpreters.append(interpreter)
            except Exception as e:
                print(f"Error loading spoof model {fname}: {e}")
                raise # Re-raise the exception to stop execution if a model fails

        if len(self.interpreters) != len(self.model_files):
             raise RuntimeError("Failed to load all required spoof detection models.")
        print(f"Spoof detection models loaded successfully from {model_dir}.")


    def detect_spoof(self, image: Image.Image, bbox: tuple) -> dict:
        """
        Detects spoof from a given PIL image and bounding box.

        Args:
            image: PIL.Image in RGB mode.
            bbox: (left, top, width, height) tuple relative to the image.

        Returns:
            A dict with keys: is_spoof (bool), score (float), time_ms (int).
        """
        outputs = []
        total_time = 0.0

        # Check if image is RGB
        if image.mode != 'RGB':
             image = image.convert('RGB')

        for scale, interpreter in zip(self.scales, self.interpreters):
            # Crop, resize, and BGR conversion (Model expects BGR)
            cropped = self._crop_and_resize(image, bbox, scale)
            arr = np.array(cropped)[..., ::-1]  # RGB -> BGR
            arr = arr.astype(np.float32)
            arr = np.expand_dims(arr, axis=0)

            # Set input and run inference
            input_details = interpreter.get_input_details()[0]
            input_index = input_details['index']

            # Ensure input type matches model expectation
            if arr.dtype != input_details['dtype']:
                arr = arr.astype(input_details['dtype'])

            interpreter.set_tensor(input_index, arr)

            start = time.time()
            interpreter.invoke()
            end = time.time()

            total_time += (end - start)

            output_index = interpreter.get_output_details()[0]['index']
            out = interpreter.get_tensor(output_index)[0]
            outputs.append(out)

        # Softmax + combine scores
        sm1 = self._softmax(outputs[0])
        sm2 = self._softmax(outputs[1])
        combined = (sm1 + sm2) / 2.0

        label = int(np.argmax(combined))
        # Assuming label 1 is 'real', other labels are 'spoof'
        is_spoof = (label != 1)
        score = float(combined[label]) # Score of the predicted class

        return {
            'is_spoof': is_spoof,
            'score': score, # Confidence score for the prediction (spoof or real)
            'spoof_label': label, # Raw label (useful for debugging)
            'combined_output': combined, # Raw combined output (useful for debugging)
            'time_ms': int(total_time * 1000)
        }

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0: return np.array([]) # Handle empty input case
        e_x = np.exp(x - np.max(x)) # Subtract max for numerical stability
        return e_x / e_x.sum()

    def _get_scaled_box(
        self,
        img_w: int,
        img_h: int,
        bbox: tuple,
        scale: float
    ) -> tuple:
        # bbox: (left, top, width, height)
        x, y, w, h = bbox
        if w <= 0 or h <= 0: # Prevent division by zero or invalid boxes
             return 0, 0, 0, 0

        # Center of the original box
        cx, cy = x + w / 2, y + h / 2

        # Calculate desired new size based on scale, but limited by image boundaries
        # This logic ensures the scaled box doesn't exceed image bounds disproportionately
        # compared to the original box's aspect ratio relative to the scale factor.
        factor_w = min(scale, (img_w - 1) / w if w > 0 else scale)
        factor_h = min(scale, (img_h - 1) / h if h > 0 else scale)
        factor = min(factor_w, factor_h) # Use the smaller factor to maintain aspect ratio within bounds

        nw, nh = w * factor, h * factor

        # Calculate new top-left and bottom-right, clamping to image dimensions
        left = max(0, int(cx - nw / 2))
        top = max(0, int(cy - nh / 2))
        right = min(img_w - 1, int(cx + nw / 2))
        bottom = min(img_h - 1, int(cy + nh / 2))

        # Ensure width/height are non-negative after clamping
        if right <= left or bottom <= top:
             return 0, 0, 0, 0

        return left, top, right, bottom

    def _crop_and_resize(
        self,
        image: Image.Image,
        bbox: tuple,
        scale: float
    ) -> Image.Image:
        img_w, img_h = image.size
        left, top, right, bottom = self._get_scaled_box(
            img_w, img_h, bbox, scale
        )

        # Check for invalid box dimensions after scaling
        if right <= left or bottom <= top:
            # Return a default small black image if the box is invalid
            print(f"Warning: Invalid bounding box after scaling: ({left}, {top}, {right}, {bottom}). Original bbox: {bbox}, scale: {scale}")
            return Image.new('RGB', (self.input_dim, self.input_dim), (0, 0, 0))

        cropped = image.crop((left, top, right, bottom))
        return cropped.resize((self.input_dim, self.input_dim), Image.Resampling.BILINEAR) # Updated Resampling method