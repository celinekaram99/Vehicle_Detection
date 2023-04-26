import cv2
import numpy as np

# Load the SSD model
model = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

# Define the classes of objects to detect
classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

# Load the input video
video = cv2.VideoCapture('1.mp4')

# Get the video frame size and fps
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

# Initialize the count and average variables
frame_count = 0
vehicle_count = 0
l = []

# Define the output video codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

# Loop over the video frames
while True:
    # Read the frame
    ret, frame = video.read()

    # If the frame was not read, break the loop
    if not ret:
        break

    # Create a blob from the input frame
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (1300, 1300), (127.5, 127.5, 127.5), False)

    # Set the input for the model
    model.setInput(blob)

    # Run the forward pass to get predictions
    detections = model.forward()

    # Count vehicles
    for i in range(detections.shape[2]):
        # Get the confidence score for the prediction
        confidence = detections[0, 0, i, 2]

        # Get the index of the class label from the detections
        class_index = int(detections[0 ,0, i, 1])

        # If the detected object is a vehicle
        if classes[class_index] in ['car','bus', 'truck']:
            # Increment the vehicle count
            vehicle_count += 1

    # Add vehicle count to list for each frame
    l.append(vehicle_count)
    vehicle_count = 0

    # Loop over the detections
    for i in range(detections.shape[2]):
        # Get the confidence score for the prediction
        confidence = detections[0, 0, i, 2]

        # Get the index of the class label from the detections
        class_index = int(detections[0, 0, i, 1])

        # If the detected object is a vehicle
        if classes[class_index] in ['car','bus', 'truck']:
            # Get the bounding box coordinates for the prediction
            x1, y1, x2, y2 = map(int, detections[0, 0, i, 3:7] * np.array([frame_width, frame_height, frame_width, frame_height]))

            # Draw the bounding box rectangle and label on
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{classes[class_index]}: {confidence:.2f}'
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, f'Vehicle count per frame: {l[-1]}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Calculate average vehicle count
    vehicle_average = sum(l) / len(l)
    cv2.putText(frame, f'Vehicle count average: {vehicle_average:.2f}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Write the output frame to the output video
    out_video.write(frame)

    # Show the output frame
    cv2.imshow('Object Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Increment frame counter
    frame_count += 1

# Release the video capture and writer objects, and close all windows
video.release()
out_video.release()
cv2.destroyAllWindows()

print(vehicle_average)