import os
import cv2

# import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils.torch_utils import select_device
from models.common import DetectMultiBackend, AutoShape

# Config value
source = "highway.mp4"
video_path = os.path.join(".", "data_ext", source)
video_path_out = os.path.join(".", "output", source)
model_path = os.path.join(".", "weights", "yolov9-c.pt")
class_names_path = os.path.join(".", "data_ext", "classes.names")

conf_threshold = 0.5
tracking_class = None

# Init Deepsort
tracker = DeepSort(max_age=70)

# Init Yolov8
device = select_device("0")  # "cuda":GPU, "cpu": CPU, "mps:0": macOS
model = DetectMultiBackend(weights=model_path, device=device, fuse=True)
model = AutoShape(model)

# Load class names from classes.names
with open(class_names_path) as f:
    class_names = f.read().strip().split("\n")

colors = [
    (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
    for j in range(len(class_names))
]
tracks = []

# Init Video capture
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

cap_out = cv2.VideoWriter(
    video_path_out,
    cv2.VideoWriter_fourcc(*"mp4v"),
    cap.get(cv2.CAP_PROP_FPS),
    (frame.shape[1], frame.shape[0]),
)

# Process every frame from video
while ret:

    # Send to model to detect
    results = model(frame)

    detections = []
    for detect_object in results.pred[0]:
        label, confidence, bbox = detect_object[5], detect_object[4], detect_object[:4]
        x1, y1, x2, y2 = map(int, bbox)
        class_id = int(label)

        if tracking_class is None:
            if confidence < conf_threshold:
                continue
        else:
            if class_id != tracking_class or confidence < conf_threshold:
                continue

        detections.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if track.is_confirmed():
            bbox = track.to_ltrb()
            x1, y1, x2, y2 = map(int, bbox)
            class_id = track.get_det_class()
            track_id = track.track_id
            color = colors[class_id]

            label = "{}-{}".format(class_names[class_id], track_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (color), 3)
            cv2.rectangle(
                frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), (color), -1
            )
            cv2.putText(
                frame,
                label,
                (x1 + 5, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

    # cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord("q"):
        break

    cap_out.write(frame)
    ret, frame = cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows()
