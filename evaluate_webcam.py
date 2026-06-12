import argparse
import numpy as np
import cv2
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from data_loader import create_generators

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

parser = argparse.ArgumentParser(description='FER evaluation & webcam inference')
parser.add_argument('--mode', type=str, default='webcam',
                    choices=['eval', 'webcam'],
                    help='eval: offline evaluation only | webcam: real-time camera (default)')
args = parser.parse_args()

print("Loading model...")
model = load_model('emotion_model.h5')
print("Model loaded successfully.\n")

if args.mode == 'eval':
    _, _, test_generator = create_generators(path='dataset', batch_size=64)

    print("=== OFFLINE EVALUATION ===")
    y_true, y_pred = [], []
    for i in range(len(test_generator)):
        x_batch, y_batch = test_generator[i]
        preds = model.predict(x_batch, verbose=0)
        y_true.extend(np.argmax(y_batch, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=EMOTIONS, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as confusion_matrix.png")
    print("Done.")
    exit()

print("=== REAL-TIME WEBCAM INFERENCE ===")
print("Press 'q' in the camera window to quit.")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
if face_cascade.empty():
    print("ERROR: Could not load Haar Cascade classifier.")
    exit()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("ERROR: Could not open webcam (index 0).")
    print("Try changing the camera index or check if another app is using it.")
    exit()

print("Webcam opened successfully.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.15, minNeighbors=5, minSize=(48, 48)
    )

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        face_resized = cv2.resize(face_roi, (48, 48))
        face_normalized = face_resized.astype(np.float32) / 255.0
        face_input = face_normalized.reshape(1, 48, 48, 1)

        preds = model.predict(face_input, verbose=0)[0]
        emotion_idx = np.argmax(preds)
        confidence = float(preds[emotion_idx])
        label = f"{EMOTIONS[emotion_idx]}: {confidence:.0%}"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Facial Expression Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam demo ended.")
