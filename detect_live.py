import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("waste_model.h5")

# Your class labels
labels = ["Metal", "Paper", "Plastic"]

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("✅ Camera opened! Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for model input
    img = cv2.resize(frame, (224, 224))
    img = img / 255.0  # normalize
    img = np.expand_dims(img, axis=0)

    # Make prediction
    preds = model.predict(img)
    label = labels[np.argmax(preds)]
    confidence = np.max(preds)

    # Show on screen
    text = f"{label} ({confidence*100:.1f}%)"
    cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 2)

    cv2.imshow("Smart Waste Detector", frame)

    # Quit if 'q' pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
