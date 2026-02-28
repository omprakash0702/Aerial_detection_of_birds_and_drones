'''from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import io

app = FastAPI()

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def home():
    return FileResponse("frontend/index.html")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

clf_model = tf.keras.models.load_model("models/bird_drone_mobilenet_final.keras")
yolo_model = YOLO("models/best.pt")

# ================= IMAGE ================= #

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...), mode: str = Form(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    if mode == "classification":
        img_resized = img.resize((224,224))
        arr = np.expand_dims(np.array(img_resized), axis=0)
        prob = clf_model.predict(arr)[0][0]

        label = "Drone" if prob > 0.5 else "Bird"
        conf = float(prob if prob>0.5 else 1-prob)

        return {"label": label, "confidence": conf}

    else:
        results = yolo_model.predict(img, conf=0.25, verbose=False)
        res = results[0]
        frame = np.array(img)

        if res.boxes is not None:
            for box in res.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = yolo_model.names[cls]

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,f"{label} {conf:.2f}",
                    (x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2)

        temp_path = "/tmp/temp_img.jpg"
        cv2.imwrite(temp_path, frame)
        return FileResponse(temp_path)

# ================= VIDEO ================= #

@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):

    input_path = "/tmp/temp_input.mp4"
    output_path = "/tmp/temp_output.mp4"

    with open(input_path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(input_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 24

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model.predict(frame, conf=0.25, verbose=False)
        res = results[0]

        if res.boxes is not None:
            for box in res.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = yolo_model.names[cls]

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,f"{label} {conf:.2f}",
                    (x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2)

        out.write(frame)

    cap.release()
    out.release()

    return FileResponse(output_path, media_type="video/mp4")

# ================= LAPTOP CAM ================= #

def generate_server_cam():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = yolo_model.predict(frame, conf=0.25, verbose=False)
        res = results[0]

        if res.boxes is not None:
            for box in res.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = yolo_model.names[cls]

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,f"{label} {conf:.2f}",
                    (x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#@app.get("/server-webcam")
def server_webcam():
    return StreamingResponse(generate_server_cam(),
        media_type='multipart/x-mixed-replace; boundary=frame')##

# ================= PHONE CAM ================= #

@app.post("/predict-frame")
async def predict_frame(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    frame = np.array(img)

    results = yolo_model.predict(frame, conf=0.25, verbose=False)
    res = results[0]

    if res.boxes is not None:
        for box in res.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = yolo_model.names[cls]

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"{label} {conf:.2f}",
                (x1,y1-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2)

    _, buffer = cv2.imencode('.jpg', frame)
    return StreamingResponse(io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg")
'''
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import io
import os

app = FastAPI()

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def home():
    return FileResponse("frontend/index.html")

# Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["ULTRALYTICS_CONFIG_DIR"] = "/tmp"

# Load models
clf_model = tf.keras.models.load_model("models/bird_drone_mobilenet_final.keras")
yolo_model = YOLO("models/best.pt")

# ================= IMAGE ================= #

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...), mode: str = Form(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    if mode == "classification":
        img_resized = img.resize((224,224))
        arr = np.expand_dims(np.array(img_resized), axis=0)
        prob = clf_model.predict(arr)[0][0]

        label = "Drone" if prob > 0.5 else "Bird"
        conf = float(prob if prob>0.5 else 1-prob)

        return {"label": label, "confidence": conf}

    else:
        # Convert PIL to NumPy (Cloud-safe)
        frame = np.array(img)

        results = yolo_model.predict(frame, conf=0.25, verbose=False)
        annotated = results[0].plot()

        _, buffer = cv2.imencode('.jpg', annotated)
        return StreamingResponse(
            io.BytesIO(buffer.tobytes()),
            media_type="image/jpeg"
        )

# ================= VIDEO ================= #

@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):

    input_path = "/tmp/temp_input.mp4"
    output_path = "/tmp/temp_output.mp4"

    with open(input_path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(input_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 24

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model.predict(frame, conf=0.25, verbose=False)
        annotated = results[0].plot()

        out.write(annotated)

    cap.release()
    out.release()

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename="result.mp4"
    )

# ================= LIVE CAM ================= #

@app.post("/predict-frame")
async def predict_frame(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    frame = np.array(img)

    results = yolo_model.predict(frame, conf=0.25, verbose=False)
    annotated = results[0].plot()

    _, buffer = cv2.imencode('.jpg', annotated)
    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg"
    )