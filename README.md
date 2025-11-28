# **📌 Bird vs Drone Classification & Object Detection**

**Deep Learning | Transfer Learning | YOLOv8 | Streamlit**

This project builds a complete AI pipeline to **classify aerial objects as Bird or Drone**, and optionally **detect** them in real-world images using YOLOv8.
The system supports **image classification, object detection, and a deployable Streamlit UI**.



## ⭐ **🚀 Project Features**

### ✅ **1. Image Classification (Binary – Bird / Drone)**

* Custom CNN baseline
* Transfer Learning:

  * EfficientNetB0
  * ResNet50V2
  * MobileNetV2 (best performer)
* Achieved up to **97% accuracy**
* Includes confusion matrix, precision, recall, and F1-score

### ✅ **2. Object Detection (YOLOv8)**

* YOLOv8n model trained on 3,319 labeled images
* Detects and labels multiple birds/drones in a single frame
* Outputs bounding boxes + class labels
* Achieved:

  * **mAP50 ≈ 0.82**
  * **Precision ≈ 0.82–0.85**
  * **Recall ≈ 0.77–0.79**

### ✅ **3. Streamlit Web App**

* Upload an image
* Select **Classification** or **YOLO Detection**
* Get predicted class + confidence score
* See bounding boxes for YOLO detection

---

## ⭐ **📂 Project Structure**


Bird_Vs_Drone/
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Custom_CNN.ipynb
│   ├── 03_Transfer_Learning.ipynb
│   ├── 04_YOLO_Training.ipynb
│
├── models/
│   ├── best_custom_cnn.h5
│   ├── best_mobilenet.keras
│   ├── best_effnetb0.keras
│   ├── best_resnet50v2.keras
│   ├── best.pt   # YOLO weights
│
├── app/
│   ├── app.py            # Streamlit App
│   ├── utils.py
│   ├── requirements.txt
│
├── object_detection_Dataset/
│   ├── train/images
│   ├── train/labels
│   ├── val/images
│   ├── val/labels
│   ├── test/images
│   ├── test/labels
│
├── classification_dataset/
│   ├── train/bird, drone
│   ├── val/bird, drone
│   ├── test/bird, drone
│
└── README.md



## ⭐ **📊 Model Performance Summary**

### 🔹 **Custom CNN**

* Accuracy: 69%
* F1-score: 0.72
* 8–10 hidden layers

### 🔹 **Transfer Learning Results**

| Model          | Accuracy | Precision | Recall | F1-score |
| -------------- | -------- | --------- | ------ | -------- |
| EfficientNetB0 | 0.972    | 0.968     | 0.968  | 0.968    |
| ResNet50V2     | 0.972    | 0.968     | 0.968  | 0.962    |
| MobileNetV2    | 0.972    | 0.968     | 0.968  | 0.968    |

> ⭐ **MobileNetV2 selected as best model for deployment** (fastest + lightest).


## ⭐ **🧠 YOLOv8 Detection Results**

* mAP50: **0.82+**
* mAP50-95: **~0.53**
* Precision: **0.82–0.85**
* Recall: **0.77–0.79**

Outputs example:

```
Bird detected at (x,y) with 0.91 confidence  
Drone detected at (x,y) with 0.87 confidence
```


## ⭐ **🖥 Streamlit App Usage**

### 🔧 **1. Install Requirements**

```
pip install -r requirements.txt
```

### ▶️ **2. Run App**

```
streamlit run app.py
```

### 🧭 **3. Features**

* Upload an image
* Choose:

  * **Classification Mode** — MobileNet prediction
  * **YOLO Detection Mode** — Bounding box detection
* View results instantly

---

## ⭐ **📘 How It Works (High Level)**

### **1. Preprocessing**

* Resize to 224×224 for classification
* Resize to 640×640 for YOLO
* Normalize pixel values
* Data augmentation applied

### **2. Model Training**

* Custom CNN baseline → moderate accuracy
* Transfer Learning → high accuracy
* YOLOv8n → bounding-box detection

### **3. Evaluation**

* Confusion matrices
* Training curves
* mAP & precision-recall metrics
* Side-by-side model comparison

### **4. Deployment**

* Streamlit web app
* Easy to upload images
* Real-time YOLO inference

---

## ⭐ **🎯 Real-World Applications**

* 🛫 Airport bird-strike prevention
* 🕊 Wildlife monitoring
* 🎥 Security & defense surveillance
* 📡 Unmanned aerial vehicle detection
* 🔬 Environmental research

---

## ⭐ **📄 Technologies Used**

* **Python, TensorFlow, Keras**
* **Transfer Learning**
* **YOLOv8 (Ultralytics)**
* **OpenCV**
* **NumPy, Pandas, Matplotlib, Seaborn**
* **Streamlit**
* **Google Colab**

---

## ⭐ **📜 License**

MIT License

---

## ⭐ **🤝 Contributing**

Pull requests are welcome.
For major changes, please open an issue first.

---

## ⭐ **💬 Contact**

Created by **Omi**
Feel free to connect or share feedback!

---

If you want, I can also generate:
✅ A professional **project thumbnail**
✅ A **requirements.txt**
✅ A **video script** for your README
Just say the word.
