from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import pickle
from scipy import ndimage
import threading
import time
import warnings
import base64
import os
# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")
# Get base directory (project root)
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


app = Flask(__name__)
app = Flask(
    __name__,
    template_folder=os.path.join(base_dir, 'templates'),
    static_folder=os.path.join(base_dir, 'static')
)
# Global variables
camera = None
current_frame = None
processed_roi = None
prediction_result = {"digit": "?", "confidence": 0, "status": "Click to start inference"}
inference_active = False
threshold = 100
frame_lock = threading.Lock()

# Load model weights at startup
try:
    with open("../Data/trained_params.pkl", "rb") as dump_file:
        W1, b1, W2, b2 = pickle.load(dump_file)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: trained_params.pkl not found in Data/")
    W1, b1, W2, b2 = None, None, None, None

# Import your model functions
try:
    from model import forward_propagation, get_predictions
except ImportError:
    print("Error: model.py not found. Make sure it's in the same directory.")
    forward_propagation, get_predictions = None, None


def NNmodel_with_confidence(X, W1, b1, W2, b2):
    """Modified NNmodel to return both prediction and confidences"""
    if forward_propagation is None or get_predictions is None:
        return 0, [0.1] * 10

    _, _, _, A2 = forward_propagation(X, W1, b1, W2, b2)
    predictions = get_predictions(A2)
    confidences = A2.flatten()  # Softmax probabilities
    return predictions, confidences


class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            raise RuntimeError("Could not start camera")

        # Set camera resolution
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def __del__(self):
        if hasattr(self, 'video'):
            self.video.release()

    def get_frame(self):
        global current_frame, processed_roi, prediction_result, threshold

        success, frame = self.video.read()
        if not success:
            return None

        with frame_lock:
            current_frame = frame.copy()

        # Convert to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create background for display
        display_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        if inference_active:
            # Apply image processing (same as your original code)
            gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Thresholding
            thresh = cv2.adaptiveThreshold(gray_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 15, 3)
            _, manual_thresh = cv2.threshold(gray_blurred, threshold, 255, cv2.THRESH_BINARY_INV)
            thresh = cv2.bitwise_and(thresh, manual_thresh)

            # Morphological operations
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            thresh = cv2.erode(thresh, kernel, iterations=1)

            # Extract center ROI (150x150)
            roi = thresh[240 - 75:240 + 75, 320 - 75:320 + 75]

            # Store processed ROI for frontend
            processed_roi = cv2.resize(roi, (100, 100), interpolation=cv2.INTER_LINEAR)

            # Process for neural network
            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            digit = None

            if contours:
                valid_contours = [
                    c for c in contours
                    if cv2.contourArea(c) > 200
                       and 30 < cv2.boundingRect(c)[2] < 120
                       and 30 < cv2.boundingRect(c)[3] < 120
                       and 0.5 < cv2.boundingRect(c)[2] / cv2.boundingRect(c)[3] < 2.0
                ]
                if valid_contours:
                    contour = max(valid_contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(contour)
                    digit = roi[y:y + h, x:x + w]
                else:
                    digit = roi
            else:
                digit = roi

            # Resize to 28x28
            resized = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_LINEAR)

            # Check if image is mostly empty
            if np.sum(resized) < 50:
                prediction_result = {
                    "digit": "?",
                    "confidence": 0,
                    "status": "No digit detected"
                }
            else:
                # Center the digit using center of mass
                cy, cx = ndimage.center_of_mass(resized)
                if not np.isnan(cx) and not np.isnan(cy):
                    shiftx = np.round(14 - cx).astype(int)
                    shifty = np.round(14 - cy).astype(int)
                    M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
                    resized = cv2.warpAffine(resized, M, (28, 28), borderValue=0)

                # Apply slight blur
                resized = cv2.GaussianBlur(resized, (3, 3), 0)

                # Normalize and predict
                normalized = resized / 255.0
                flat = normalized.reshape(784, 1)

                if W1 is not None and forward_propagation is not None:
                    prediction, confidences = NNmodel_with_confidence(flat, W1, b1, W2, b2)
                    predicted_digit = int(prediction.item())
                    confidence = float(confidences[predicted_digit] * 100)

                    prediction_result = {
                        "digit": predicted_digit,
                        "confidence": round(confidence, 1),
                        "status": "Prediction active"
                    }
                else:
                    prediction_result = {
                        "digit": "?",
                        "confidence": 0,
                        "status": "Model not loaded"
                    }

            # Create display frame with threshold visualization
            display_frame = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

            # Draw ROI rectangle
            cv2.rectangle(display_frame, (320 - 75, 240 - 75), (320 + 75, 240 + 75), (0, 255, 0), 2)

            # Add prediction text overlay
            cv2.putText(display_frame, f"Digit: {prediction_result['digit']}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Confidence: {prediction_result['confidence']}%", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        else:
            # Show normal camera feed when inference is off
            display_frame = frame
            # Add instructions
            cv2.putText(display_frame, "Click 'Start Inference' to begin", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "Adjust threshold with slider", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            prediction_result = {
                "digit": "?",
                "confidence": 0,
                "status": "Click to start inference"
            }

        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', display_frame)
        frame_bytes = buffer.tobytes()
        return frame_bytes


def get_camera():
    global camera
    if camera is None:
        try:
            camera = VideoCamera()
        except RuntimeError as e:
            print(f"Camera error: {e}")
            return None
    return camera


def generate_frames():
    camera = get_camera()
    if camera is None:
        # Return a black frame with error message
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Camera not available", (200, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return

    while True:
        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.1)  # Limit frame rate


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/toggle_inference', methods=['POST'])
def toggle_inference():
    global inference_active
    inference_active = not inference_active
    return jsonify({
        'inference_active': inference_active,
        'status': 'Inference started' if inference_active else 'Inference stopped'
    })


@app.route('/set_threshold', methods=['POST'])
def set_threshold():
    global threshold
    data = request.get_json()
    threshold = int(data.get('threshold', 100))
    return jsonify({'threshold': threshold})


@app.route('/get_roi')
def get_roi():
    global processed_roi
    if processed_roi is not None:
        ret, buffer = cv2.imencode('.jpg', processed_roi)
        roi_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'roi_image': f'data:image/jpeg;base64,{roi_base64}'})
    else:
        return jsonify({'roi_image': None})


@app.route('/get_prediction')
def get_prediction():
    return jsonify(prediction_result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)