import cv2
import time
import os

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  

output_directory = "data/down"
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
while (time.time() - start_time) < 2:
    ret, frame = cap.read()
    if not ret:
        break

    out.write(frame)

    cv2.imshow("Captured Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()

cv2.destroyAllWindows()