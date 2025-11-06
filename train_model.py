import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

dataset_path = "waste_dataset"
categories = ['Plastic', 'Paper', 'Metal']
img_size = 64

data, labels = [], []

for idx, cat in enumerate(categories):
    folder = os.path.join(dataset_path, cat)
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue
        img = cv2.resize(img, (img_size, img_size))
        data.append(img)
        labels.append(idx)

data = np.array(data, dtype=np.float32) / 255.0
labels = to_categorical(labels, num_classes=len(categories))

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'), MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'), MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'), Dropout(0.5),
    Dense(len(categories), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=16)

model.save('waste_model.h5')
print("Model trained and saved as waste_model.h5")
