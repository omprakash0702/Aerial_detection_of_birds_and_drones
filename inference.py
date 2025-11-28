from ultralytics import YOLO
import cv2
import os

# ==== CONFIG ====
YOLO_MODEL_PATH = "models/best.pt"   # same model as your Streamlit app
INPUT_VIDEO = "2.mp4"            # path to your test video
OUTPUT_VIDEO = "output_yolo.mp4"     # path where annotated video will be saved


def main():
    # Load YOLO model
    model = YOLO(YOLO_MODEL_PATH)

    # Open video
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"❌ Could not open video: {INPUT_VIDEO}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define VideoWriter for output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    frame_idx = 0
    print("⚙️ Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Run YOLO detection on this frame
        results = model.predict(
            frame,
            imgsz=640,
            conf=0.25,
            verbose=False
        )

        # Get annotated frame (BGR)
        annotated = results[0].plot()

        # Write annotated frame to output video
        out.write(annotated)

        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx} frames...")

    cap.release()
    out.release()
    print(f"✅ Done. Saved annotated video to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
