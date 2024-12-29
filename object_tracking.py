import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils.torch_utils import select_device
from models.common import DetectMultiBackend, AutoShape

# Config value
video_path = "data_ext/high.mp4"
conf_threshhold = 0.5
tracking_class = 2

# Init Deepsort 
tracker = DeepSort(max_age=5)

# Init Yolov8
device = select_device('cpu') # "cuda":GPU, "cpu": CPU, "mps:0": macOS
model = DetectMultiBackend(weights="weights/yolov8n.pt", device=device, fuse=True)
model  = AutoShape(model)

# Load class names from classes.names
with open("data_ext/classes.names") as f:
    class_names = f.read().strip().split('\n')

color = np.random.randint(0,255, size=(len(class_names),3))
tracks = []

# Init Video capture
cap = cv2.VideoCapture(video_path)

# Process every frame from video
while True:
    
    # Read
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Send to model to detect
    results = model(frame)
    detect = []
    for dectected_object in results.pred[0] :
        label, confidence, bbox = dectected_object[5], dectected_object[4], dectected_object[:4]
        x1, y1, x2, y2 = map(int, bbox)
        class_id = int(label)

        if tracking_class is None :
            if confidence < conf_threshhold :
                continue
        else :
            if class_id != tracking_class or confidence < conf_threshhold :
                continue
        
        detect.append([[x1,y1,x2,y2], confidence, class_id])
    
    # Update, Indexing with Deepsort
    tracks = tracker.update_tracks(detect, frame=frame)
    
    # Vẽ lên màn hình các khung chữ nhật kèm ID
    for track in tracks:
        if track.is_confirmed():
            track_id = track.track_id

            # Lấy toạ độ, class_id để vẽ lên hình ảnh
            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)
            color = colors[class_id]
            B, G, R = map(int,color)

            label = "{}-{}".format(class_names[class_id], track_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
            cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), (B, G, R), -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Show hình ảnh lên màn hình
    cv2.imshow("OT", frame)
    # Bấm Q thì thoát
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
    
    