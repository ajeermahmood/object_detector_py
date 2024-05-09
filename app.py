from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np

# from flask_cors import CORS
import base64
from PIL import Image
import io

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret00082#2%!"
socketio = SocketIO(app)
socketio.init_app(app, cors_allowed_origins="*")
# CORS(app, origins="http://localhost:4200")


# Load YOLO
net = cv2.dnn.readNet("assets/yolov3.weights", "assets/yolov3.cfg")
classes = []
with open("assets/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


# Function to perform object detection on an image
def detect_objects(image):
    # Resize and normalize image
    img = cv2.resize(image, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    # Detect objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process detections
    class_names_detected = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

                if classes[class_id] not in class_names_detected:
                    class_names_detected.append(classes[class_id])

    return boxes, confidences, class_names_detected


# Take in base64 string and return PIL image
def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))


# convert PIL Image to an RGB image( technically a numpy array ) that's compatible with opencv
def toRGB(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)


@socketio.on("connect")
def handle_connect():
    origin = request.headers.get("Origin")
    print("New connection from origin:", origin)

    emit("connected", "Successfully Connected")


@socketio.on("image")
def handle_image(data):
    print("getting data.....")
    # Decode base64 image
    image_decoded = stringToImage(data)
    img_colored = toRGB(image_decoded)

    # Perform object detection
    boxes, confidences, items = detect_objects(img_colored)

    # Emit detected objects
    emit(
        "detected_objects", {"boxes": boxes, "confidences": confidences, "items": items}
    )


if __name__ == "__main__":
    socketio.run(app, debug=True, host="localhost")
