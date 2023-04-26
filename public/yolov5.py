import cv2
import torch
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Open video file
video_path = '1.mp4'
cap = cv2.VideoCapture(video_path)

# Initialize frame counter and vehicle count
frame_count = 0
vehicle_count = 0
l = []

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# Loop through video frames
while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection with YOLOv5
    results = model(frame)
    detections = results.xyxy[0].tolist()

    # Count vehicles
    for detection in detections:
        if detection[5] == 2 or detection[5] == 3 or detection[5] == 5 or detection[5] == 7:
            vehicle_count += 1
    
    # Add vehicle count to list for each frame
    l.append(vehicle_count)
    vehicle_count = 0

    # Draw bounding boxes and vehicle count on frame
    for detection in detections:
        bbox = detection[:4]
        label = int(detection[5])
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.putText(frame, f'{label}', (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, f'Vehicle count per frame: {l[-1]}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Calculate average vehicle count
    vehicle_average = sum(l) / len(l)
    cv2.putText(frame, f'Vehicle count average: {vehicle_average:.2f}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Write frame to output video
    out.write(frame)

    # Display frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

    # Increment frame counter
    frame_count += 1

# Release video file, video writer and close window
cap.release()
out.release()
cv2.destroyAllWindows()

print(vehicle_average)