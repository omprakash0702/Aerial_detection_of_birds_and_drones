import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io

from ultralytics import YOLO


# =============================
# CONFIG
# =============================

IMG_SIZE = (224, 224)  # for classifier

# MUST match your train_gen.class_indices order
CLASS_INDICES = {"bird": 0, "drone": 1}
IDX_TO_CLASS = {v: k for k, v in CLASS_INDICES.items()}

CLASS_MODEL_PATH = "models/bird_drone_mobilenet_final.keras"
YOLO_MODEL_PATH = "models/best.pt"  # put your YOLO best.pt here


# =============================
# MODEL LOADERS (CACHED)
# =============================

@st.cache_resource
def load_classification_model():
    model = tf.keras.models.load_model(CLASS_MODEL_PATH)
    return model


@st.cache_resource
def load_yolo_model():
    model = YOLO(YOLO_MODEL_PATH)
    return model


# =============================
# PREPROCESSING & PREDICTION
# =============================

def preprocess_image_for_classifier(pil_img: Image.Image) -> np.ndarray:
    """Resize + convert PIL image to model-ready numpy array."""
    pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize(IMG_SIZE)
    arr = img_to_array(pil_img)
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)

    # IMPORTANT: if you added preprocess_input INSIDE the model during training,
    # then DO NOT rescale or preprocess again here.
    return arr


def predict_class(model, pil_img: Image.Image, threshold: float = 0.5):
    arr = preprocess_image_for_classifier(pil_img)
    prob = model.predict(arr)[0][0]  # scalar: p(drone)

    pred_label = 1 if prob > threshold else 0
    class_name = IDX_TO_CLASS[pred_label]
    confidence = prob if pred_label == 1 else (1 - prob)
    return class_name, float(confidence), float(prob)


def run_yolo_detection(yolo_model, pil_img: Image.Image, conf: float = 0.25):
    """
    Run YOLOv8 on a PIL image.
    Returns:
      - annotated PIL image
      - list of detections: (class_name, confidence)
    """
    # YOLO can work directly with a PIL image
    results = yolo_model.predict(
        pil_img,
        imgsz=640,
        conf=conf,
        verbose=False
    )

    res = results[0]

    # Annotated image as numpy (BGR) -> convert to RGB for Streamlit
    annotated_bgr = res.plot()  # numpy array, BGR channel order
    annotated_rgb = annotated_bgr[..., ::-1]  # BGR -> RGB
    annotated_pil = Image.fromarray(annotated_rgb)

    dets = []
    if res.boxes is not None and len(res.boxes) > 0:
        for box in res.boxes:
            cls_idx = int(box.cls[0].item())
            score = float(box.conf[0].item())
            cls_name = res.names.get(cls_idx, str(cls_idx))
            dets.append((cls_name, score))

    return annotated_pil, dets


# =============================
# STREAMLIT UI
# =============================

st.set_page_config(
    page_title="Bird vs Drone – Classifier & Detector",
    page_icon="🦅",
    layout="centered"
)

st.title("🦅 Bird vs 🤖 Drone – Aerial AI")
st.write(
    "Upload aerial images and choose between **classification** (CNN) and **object detection** (YOLOv8)."
)

# Load models once
clf_model = load_classification_model()
yolo_model = load_yolo_model()

tab_cls, tab_yolo = st.tabs(["📌 Classification (CNN)", "🎯 Detection (YOLOv8)"])


# =============================
# TAB 1: CLASSIFICATION
# =============================

with tab_cls:
    st.subheader("Image Classification – Custom CNN + MobileNetV2")

    uploaded_file_cls = st.file_uploader(
        "Upload an image (JPG/PNG) for classification",
        type=["jpg", "jpeg", "png"],
        key="cls_uploader"
    )

    if uploaded_file_cls is not None:
        image_bytes = uploaded_file_cls.read()
        pil_img_cls = Image.open(io.BytesIO(image_bytes))

        st.image(pil_img_cls, caption="Uploaded image", use_container_width=True)

        with st.spinner("Running classification..."):
            label, conf, raw_prob = predict_class(clf_model, pil_img_cls, threshold=0.5)

        st.markdown("### Prediction (Classifier)")
        pretty_label = "Bird 🐦" if label == "bird" else "Drone 🤖"
        st.write(f"**Class:** {pretty_label}")
        st.write(f"**Confidence:** {conf:.2%}")
        st.caption(f"Raw p(drone) = {raw_prob:.4f}")

        st.progress(conf if label == "drone" else 1 - conf)

    else:
        st.info("👆 Upload an image above to classify it as **Bird** or **Drone**.")


# =============================
# TAB 2: YOLO DETECTION
# =============================

with tab_yolo:
    st.subheader("Object Detection – YOLOv8n (Bird vs Drone)")

    uploaded_file_det = st.file_uploader(
        "Upload an image (JPG/PNG) for detection",
        type=["jpg", "jpeg", "png"],
        key="det_uploader"
    )

    conf_thres = st.slider(
        "Detection confidence threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.25,
        step=0.05
    )

    if uploaded_file_det is not None:
        image_bytes = uploaded_file_det.read()
        pil_img_det = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        st.image(pil_img_det, caption="Original image", use_container_width=True)

        with st.spinner("Running YOLOv8 detection..."):
            annotated_pil, detections = run_yolo_detection(
                yolo_model, pil_img_det, conf=conf_thres
            )

        st.markdown("### Detections")
        if len(detections) == 0:
            st.warning("No birds or drones detected above the confidence threshold.")
        else:
            for cls_name, score in detections:
                st.write(f"- **{cls_name}** with confidence **{score:.2%}**")

        st.markdown("### Annotated Image")
        st.image(annotated_pil, caption="YOLOv8 Detections", use_container_width=True)
    else:
        st.info("👆 Upload an image to run YOLOv8 detection on it.")
