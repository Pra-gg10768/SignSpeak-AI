import cv2
import numpy as np
from tensorflow.keras.models import load_model
from gtts import gTTS
import os
import uuid
import playsound

# Try importing cv2_imshow for Colab
try:
    from google.colab.patches import cv2_imshow
    USE_COLAB = True
except ImportError:
    USE_COLAB = False

# Load trained model
model = load_model('sign_language_model.h5')

# Gesture labels (A–Z excluding J and Z)
gesture_labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
    'T', 'U', 'V', 'W', 'X', 'Y'
]

# Text-to-Speech using gTTS
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    filename = f"{uuid.uuid4()}.mp3"
    tts.save(filename)
    playsound.playsound(filename)
    os.remove(filename)

# Preprocess webcam frame for model
def preprocess_image(image, size=(64, 64)):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, size)
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    return np.expand_dims(img, axis=0)  # Add batch dimension

# Start webcam
cap = cv2.VideoCapture(0)
last_prediction = ""
frame_count = 0

print("🟢 Starting SignSpeak-AI. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to read from webcam.")
        break

    # Region of interest (ROI) where hand is expected
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]

    # Preprocess and predict
    processed = preprocess_image(roi)
    prediction = model.predict(processed)
    predicted_class = np.argmax(prediction)
    gesture_text = gesture_labels[predicted_class]

    # Draw ROI and prediction
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f'Gesture: {gesture_text}', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Speak if gesture changes and is stable
    if gesture_text != last_prediction:
        frame_count += 1
        if frame_count >= 5:
            print(f"🔊 Speaking: {gesture_text}")
            text_to_speech(gesture_text)
            last_prediction = gesture_text
            frame_count = 0
    else:
        frame_count = 0

    # Show frame
    if USE_COLAB:
        cv2_imshow(frame)
    else:
        cv2.imshow("SignSpeak-AI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Quitting...")
        break

cap.release()

# Safe close for OpenCV GUI
try:
    cv2.destroyAllWindows()
except:
    pass
