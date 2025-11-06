import cv2
import os
import time

# Set up dataset folder
DATASET_PATH = "waste_dataset"
CLASSES = ["Plastic", "Paper", "Metal"]

for name in CLASSES:
    os.makedirs(os.path.join(DATASET_PATH, name), exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not found!")
    exit()

print("Camera opened successfully!")
print("\nInstructions:")
print("1 - Save as Plastic")
print("2 - Save as Paper")
print("3 - Save as Metal")
print("q - Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    cv2.putText(frame, "Press 1: Plastic | 2: Paper | 3: Metal | q: Quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Dataset Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key in [ord('1'), ord('2'), ord('3')]:
        label = CLASSES[int(chr(key)) - 1]
        filename = os.path.join(DATASET_PATH, label,
                                f"{label}_{int(time.time())}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")

cap.release()
cv2.destroyAllWindows()
