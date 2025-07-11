!pip install ultralytics flask flask-cors opencv-python pyngrok waitress

from ultralytics import YOLO
import cv2
from flask import Flask, Response
from flask_cors import CORS
from pyngrok import ngrok
from waitress import serve

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load YOLO model
model = YOLO("yolov8n.pt")  # Use YOLOv8 Nano for efficiency

# Initialize camera
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # Run detection
        results = model(frame, conf=0.6, classes=[67])  # Class 67 is 'cell phone'

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()

                # Draw detection
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, f"PHONE DETECTED: {conf:.2f}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                # Add warning overlay
                cv2.putText(frame, "WARNING: DON'T USE PHONE WHILE DRIVING!",
                           (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                frame = cv2.addWeighted(frame, 0.7,
                                      cv2.rectangle(frame.copy(), (0,0), (frame.shape[1],80),
                                      (0,0,0), -1), 0.3, 0)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return """
    <html>
      <head><title>Phone Detection</title></head>
      <body>
        <h1>Phone Detection While Driving</h1>
        <img src="/video_feed" width="800">
      </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Ngrok setup
ngrok.set_auth_token("2zUdrhZIUzWrNG4zKEm7Tgbes6F_dpPWmZpi24FpkQ9TUf2e")
tunnel = ngrok.connect(addr="5000", proto="http", bind_tls=True)
print(f"Public URL: {tunnel.public_url}")

# Start server
serve(app, host='0.0.0.0', port=5000)


from ultralytics import YOLO
import cv2
from IPython.display import display, Image
import time

model = YOLO("yolov8n.pt")  # Tiny model for speed
cap = cv2.VideoCapture(1)   # Webcam

while True:
    ret, frame = cap.read()
    if not ret: break

    # Only process every 3rd frame (reduces load)
    if int(time.time() * 10) % 3 == 0:
        results = model(frame, classes=[67], conf=0.6)  # Class 67 = phone
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Show in Colab (no Flask/ngrok)
    _, buf = cv2.imencode('.jpg', frame)
    display(Image(data=buf.tobytes()))
    time.sleep(0.1)  # ~10 FPS limit








from ultralytics import YOLO
import cv2
from IPython.display import display, clear_output
import time

# Load YOLO model
model = YOLO("yolov8n.pt")  # Use the nano model for speed

# Upload a test video
from google.colab import files
uploaded = files.upload()

from IPython.display import Image
from google.colab.patches import cv2_imshow

# Use uploaded video
video_path = list(uploaded.keys())[0]
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detection
results = model(frame, conf=0.3)
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = model.names[cls]
        conf = box.conf[0].item()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)



    # Show frame
    _, buffer = cv2.imencode('.jpg', frame)
    display(Image(data=buffer))
    clear_output(wait=True)
    time.sleep(0.1)

     # Show result
    cv2_imshow(frame)

cap.release()







