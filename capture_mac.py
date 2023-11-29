import cv2
import time
import os

# Use '0' for the default camera on macOS
capture_device_type = cv2.CAP_AVFOUNDATION
cap = cv2.VideoCapture(capture_device_type)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set camera resolution to 640x480
cap.set(3, 640)
cap.set(4, 480)

# Use 'avc1' as the fourcc for macOS (QuickTime)
fourcc = cv2.VideoWriter_fourcc(*'avc1')

output_directory = "data/up"
os.makedirs(output_directory, exist_ok=True)

counter = 1
output_filename = os.path.join(output_directory, f"video_{counter}.mp4")
while os.path.isfile(output_filename):
    counter += 1
    output_filename = os.path.join(output_directory, f"video_{counter}.mp4")

output_fps = 30
output_width = int(cap.get(3)) 
output_height = int(cap.get(4))
out = cv2.VideoWriter(output_filename, fourcc, output_fps, (output_width, output_height))

start_time = time.time()

num_frames = 0
# while (time.time() - start_time) < 2:
while num_frames < 49:
    ret, frame = cap.read()
    if not ret:
        break

    out.write(frame)

    cv2.imshow("Captured Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    num_frames += 1

cap.release()
out.release()

cv2.destroyAllWindows()
