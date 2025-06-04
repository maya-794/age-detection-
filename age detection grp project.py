import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
# Load the trained CNN model
# model = tf.keras.models.load_model("age_model.h5", compile=False)
model = tf.keras.models.load_model(r"C:\Users\srima\age_model.h5",compile=False)
# model = build_age_model()
# model.load_weights("age_model.h5")  # Uncomment to load pre-trained weights

# Define age categories
age_classes = ["0-12", "13-20", "21-35", "36-50", "51-60","60+"]

# Initialize OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y + h, x:x + w]
        face_img = cv2.resize(face_img, (64, 64))
        face_img = img_to_array(face_img) / 255.0
        face_img = np.expand_dims(face_img, axis=0)

        prediction = model.predict(face_img)
        age_label = age_classes[np.argmax(prediction)]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(frame, age_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Age Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


