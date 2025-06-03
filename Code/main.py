import cv2
import numpy as np
import pickle
from scipy import ndimage
from model import NNmodel, forward_propagation, get_predictions  # Import necessary functions
import os
import warnings

# Suppress SciPy warning for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load model weights
try:
    with open("../Data/trained_params.pkl", "rb") as dump_file:
        W1, b1, W2, b2 = pickle.load(dump_file)
except FileNotFoundError:
    print("Error: trained_params.pkl not found in ../Data/")
    exit(1)

# Global control variables
startInference = False
threshold = 100  # Default threshold
frameCount = 0
showDebugWindows = True  # Toggle for additional debug windows

# Mouse toggle for inference
def ifClicked(event, x, y, flags, params):
    global startInference
    if event == cv2.EVENT_LBUTTONDOWN:
        startInference = not startInference

# Trackbar control
def on_threshold(x):
    global threshold
    threshold = x

# Modified NNmodel to return both prediction and confidences
def NNmodel_with_confidence(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_propagation(X, W1, b1, W2, b2)
    predictions = get_predictions(A2)
    confidences = A2.flatten()  # Softmax probabilities
    return predictions, confidences

# Start camera + OpenCV window
def start_cv():
    global threshold, frameCount

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Check webcam resolution
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_width < 640 or frame_height < 480:
        print(f"Warning: Webcam resolution {frame_width}x{frame_height} is too low. Minimum required: 640x480")
        cap.release()
        return

    cv2.namedWindow('background')
    cv2.setMouseCallback('background', ifClicked)
    cv2.createTrackbar('threshold', 'background', threshold, 255, on_threshold)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        # Convert to grayscale (do this once per frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        background = np.zeros((480, 640), dtype=np.uint8)

        if startInference:
            frameCount += 1

            # Apply slight blur to reduce noise
            gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Combine adaptive and binary thresholding with tuned parameters
            thresh = cv2.adaptiveThreshold(gray_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 15, 3)
            _, manual_thresh = cv2.threshold(gray_blurred, threshold, 255, cv2.THRESH_BINARY_INV)
            thresh = cv2.bitwise_and(thresh, manual_thresh)

            # Apply morphological operations to clean up noise
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            thresh = cv2.erode(thresh, kernel, iterations=1)  # Additional erosion to remove small speckles

            # Extract center ROI (150x150) ( Region of Intrest )
            roi = thresh[240-75:240+75, 320-75:320+75]
            background[240-75:240+75, 320-75:320+75] = roi

            # Optional: Show raw thresholded ROI for debugging
            if showDebugWindows:
                cv2.imshow('Thresholded ROI', cv2.resize(roi, (150, 150), interpolation=cv2.INTER_LINEAR))

            # Find bounding box of digit
            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            digit = None
            if contours:
                # Filter contours by area and aspect ratio
                valid_contours = [
                    c for c in contours
                    if cv2.contourArea(c) > 200
                    and 30 < cv2.boundingRect(c)[2] < 120
                    and 30 < cv2.boundingRect(c)[3] < 120
                    and 0.5 < cv2.boundingRect(c)[2]/cv2.boundingRect(c)[3] < 2.0
                ]
                if valid_contours:
                    contour = max(valid_contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(contour)
                    digit = roi[y:y+h, x:x+w]
                else:
                    digit = roi
            else:
                digit = roi

            # Resize to 28x28 with smoother interpolation
            resized = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_LINEAR)

            # Check if the image is all black
            if np.sum(resized) < 50:  # Relaxed threshold for "empty" check
                cv2.putText(background, "No digit detected", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                cv2.putText(background, "Try lowering threshold", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                # Center the digit using center of mass
                cy, cx = ndimage.center_of_mass(resized)
                if not np.isnan(cx) and not np.isnan(cy):
                    shiftx = np.round(14 - cx).astype(int)
                    shifty = np.round(14 - cy).astype(int)
                    M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
                    resized = cv2.warpAffine(resized, M, (28, 28), borderValue=0)

                # Apply slight blur to reduce edge artifacts
                resized = cv2.GaussianBlur(resized, (3, 3), 0)

                # Normalize to [0, 1]
                normalized = resized / 255.0
                flat = normalized.reshape(784, 1)

                # Run model prediction with confidences
                prediction, confidences = NNmodel_with_confidence(flat, W1, b1, W2, b2)
                predicted_digit = int(prediction.item())

                # Display prediction and confidence
                cv2.putText(background, f"Digit: {predicted_digit}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
                top_conf = confidences[predicted_digit] * 100
                cv2.putText(background, f"Confidence: {top_conf:.1f}%", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Draw ROI rectangle
            cv2.rectangle(background, (320-75, 240-75), (320+75, 240+75), (255, 255, 255), 2)

            # Show processed digit with a grid for better assessment
            processed_display = cv2.resize(resized, (100, 100), interpolation=cv2.INTER_LINEAR)
            # Add grid lines (every 25 pixels, since 100/4 = 25)
            for i in range(1, 4):
                cv2.line(processed_display, (i*25, 0), (i*25, 100), (128, 128, 128), 1)
                cv2.line(processed_display, (0, i*25), (100, i*25), (128, 128, 128), 1)
            cv2.imshow('Processed Digit', processed_display)

        else:
            background = gray
            # Add instructions when inference is off
            cv2.putText(background, "Click to start inference", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.putText(background, "Adjust threshold with slider", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('background', background)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_cv()