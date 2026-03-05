
---

# 🐦 Bird vs 🤖 Drone Detection & Classification System

**Deep Learning | Transfer Learning | YOLOv8 | FastAPI**

An end-to-end computer vision system that classifies aerial objects as **Bird** or **Drone** and detects them in real-world images, videos, and live camera feeds.
The project combines **transfer learning for classification** and **YOLOv8 object detection** with a **FastAPI backend and lightweight web interface** for interactive inference.

The system supports:

* Image classification
* Image object detection
* Video object detection
* Laptop webcam live detection
* Mobile camera real-time detection via browser

---

# 🚀 Project Features

## 1️⃣ Image Classification (Bird vs Drone)

Binary classification using deep learning and transfer learning.

Models explored:

* Custom CNN baseline
* **EfficientNetB0**
* **ResNet50V2**
* **MobileNetV2 (best performer)**

### Performance

| Model           | Accuracy  | Precision | Recall    | F1-score  |
| --------------- | --------- | --------- | --------- | --------- |
| Custom CNN      | 0.69      | 0.70      | 0.71      | 0.72      |
| EfficientNetB0  | 0.972     | 0.968     | 0.968     | 0.968     |
| ResNet50V2      | 0.972     | 0.968     | 0.968     | 0.962     |
| **MobileNetV2** | **0.972** | **0.968** | **0.968** | **0.968** |

📌 **MobileNetV2 selected for deployment** due to its speed and lightweight architecture.

The classifier outputs:

* Predicted class (Bird / Drone)
* Confidence score

---

# 🎯 Object Detection (YOLOv8)

Object detection is implemented using **YOLOv8n**, trained on a labeled dataset of aerial bird and drone images.

### Dataset

* **3,319 labeled images**
* Bounding box annotations
* Bird and Drone classes

### Detection Performance

| Metric    | Score     |
| --------- | --------- |
| mAP50     | ~0.82     |
| mAP50–95  | ~0.53     |
| Precision | 0.82–0.85 |
| Recall    | 0.77–0.79 |

The detector identifies multiple objects in a frame and outputs:

* Bounding boxes
* Class labels
* Confidence scores

Example output:

```
Bird detected at (x1, y1, x2, y2) with 0.91 confidence
Drone detected at (x1, y1, x2, y2) with 0.87 confidence
```

---

# 🖥 Web Interface (FastAPI + HTML/CSS)

The project includes a lightweight web interface powered by **FastAPI** and a simple **HTML/CSS/JavaScript frontend**.

The interface allows users to interact with the models without needing Python knowledge.

### Supported Inference Modes

### 📷 Image Upload

Users can upload an image and choose:

* **Classification Mode**
* **YOLO Detection Mode**

The system returns predictions along with bounding boxes if detection is selected.

---

### 🎥 Video Upload

Users can upload a video file.

The backend processes the video frame-by-frame using YOLOv8 and returns a **fully annotated video with detections**.

---

### 💻 Laptop Webcam Detection

Live object detection can be performed using the system's webcam.

Each frame is processed by YOLOv8 and displayed in real time with bounding boxes.

---

### 📱 Mobile Camera Detection

The interface also supports **mobile phone cameras via the browser**.

Using the device camera:

* Frames are captured in the browser
* Sent to the FastAPI backend
* YOLOv8 performs inference
* Annotated results are streamed back

This allows real-time aerial object detection directly from a smartphone.

---

# 🧠 How the System Works

### 1. Data Preprocessing

For classification:

* Resize images to **224 × 224**
* Normalize pixel values
* Apply data augmentation

For object detection:

* Resize images to **640 × 640**
* Convert annotations to YOLO format
* Perform augmentation

---

### 2. Model Training

**Classification Pipeline**

1. Custom CNN baseline
2. Transfer learning models
3. Hyperparameter tuning
4. Evaluation with confusion matrix

**Object Detection Pipeline**

1. YOLOv8 training
2. Bounding box learning
3. mAP evaluation
4. Inference testing

---

### 3. Inference Pipeline

For each request:

1. Input received from UI
2. Preprocessing applied
3. Model inference executed
4. Output returned

Outputs include:

* Predicted class
* Confidence score
* Bounding boxes
* Annotated images or videos

---



---

# 🧰 Technologies Used

### Machine Learning

* TensorFlow / Keras
* Transfer Learning
* YOLOv8 (Ultralytics)

### Computer Vision

* OpenCV
* NumPy
* PIL

### Backend

* FastAPI
* Python

### Frontend

* HTML
* CSS
* JavaScript

### Tools

* Google Colab
* Jupyter Notebook

---

# 🎯 Real-World Applications

### ✈ Airport Safety

Detect drones and birds near airport runways to reduce bird-strike risks.

### 🛰 Surveillance Systems

Monitor unauthorized drones in restricted airspace.

### 🐦 Wildlife Monitoring

Track bird activity for ecological research.

### 🛡 Security & Defense

Identify aerial objects in surveillance footage.

### 📡 UAV Monitoring

Assist in drone traffic monitoring systems.

---

# 📊 Example Outputs

### Classification Output

```
Prediction: Bird
Confidence: 97.3%
```

---

### Detection Output

Bounding boxes with labels:

```
Bird — 0.91
Drone — 0.87
```

---

# 📜 License

MIT License

---

# 🤝 Contributing

Contributions are welcome.

If you'd like to improve the system:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

# 👨‍💻 Author

**Omprakash**

If you found this project useful, feel free to connect or share feedback.

---

If you want, I can also generate:

✅ **A professional GitHub project banner**
✅ **Screenshots section for README**
✅ **Short demo video script**
✅ **Better GitHub repo structure (industry-style)**

Those make the project **much stronger for ML/AI resumes.**
