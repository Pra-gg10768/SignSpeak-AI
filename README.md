# 🤟 SignSpeak-AI

**SignSpeak-AI** is an AI-powered real-time application that translates **sign language gestures** into **text and speech** using a **CNN-LSTM deep learning model**, **OpenCV**, and **text-to-speech (TTS)**. It's designed to bridge the communication gap between the hearing and hearing-impaired communities.

---

## 🌟 Features

- 🖐️ Real-time Sign Language Recognition via Webcam
- 🔊 Speech Output using Google Text-to-Speech (gTTS)
- 🧠 CNN + LSTM model trained on ASL dataset (e.g., Sign Language MNIST)
- 📦 Built with TensorFlow, OpenCV, NumPy, and gTTS
- 🗣️ Future-ready for multilingual and voice-to-sign translation (planned)

---

##🧠 How It Works
SignSpeak-AI is a real-time gesture-to-speech application that turns hand signs into spoken words. Here's a breakdown of how everything flows behind the scenes:

-🖐️ 1. Gesture Capturing via Webcam
The app opens your camera using OpenCV.

A green ROI (Region of Interest) appears on screen.

You show a hand sign inside this box (e.g., letter 'A').

ret, frame = cap.read()
roi = frame[y1:y2, x1:x2]

-🧼 2. Image Preprocessing
The hand gesture is processed to match the format expected by the AI model:

Converted to grayscale

Resized to 64x64 pixels

Normalized to scale values between 0 and 1

Reshaped to fit the model input format

img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (64, 64))
img = img / 255.0
img = np.expand_dims(img, axis=-1)  # Channel dimension
img = np.expand_dims(img, axis=0)   # Batch dimension

-🤖 3. Prediction with CNN Model
The preprocessed image is passed into a trained CNN model. The model outputs a prediction for one of the 24 supported ASL letters (A–Y, excluding J and Z).

prediction = model.predict(processed)
gesture_text = gesture_labels[np.argmax(prediction)]

-🔊 4. Voice Output (TTS)
Once a stable and changed gesture is detected, the system:

Displays the gesture label on screen

Converts it to speech using a TTS engine like pyttsx3 or gTTS

engine.say(gesture_text)
engine.runAndWait()

-🔁 5. Continuous Loop
This process runs continuously in real-time until you press the q key to quit.

🤔 Why J and Z Are Excluded?
ASL letters J and Z require motion-based gestures (e.g., tracing a letter in the air). Since this system uses static images only, it does not currently support motion gestures.

📊 System Pipeline

[ You ] 
  ↓ (hand gesture)
[ Webcam ROI Captured ] 
  ↓
[ Image Preprocessing ]
  ↓
[ Trained CNN Model ]
  ↓
[ Predicted Gesture ]
  ↓
[ Text Display + Speech Output ]
