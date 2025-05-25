import cv2
import mediapipe as mp
import numpy as np
import joblib
import pygame
import os
import pandas as pd

# Load model and label encoder
model = joblib.load("models/gesture_classifier.pkl")
label_encoder = joblib.load("models/gesture_labels.pkl")

# Initialize pygame mixer
pygame.mixer.init()

# Load notes (assuming notes/A.wav, notes/B.wav, etc.)
notes_folder = "notes"
note_names = ["A", "B", "C", "D", "E", "F", "G"]
note_sounds = {name: pygame.mixer.Sound(os.path.join(notes_folder, f"{name}.wav")) for name in note_names}

# Map gesture labels to note names
gesture_to_note = {
    "peace_sign": "C",
    "fist": "D",
    "thumbs_up": "E",
    "open_palm": "F",
    "call_me": "G",
    "index_finger": "A",
    "rock_on": "B"
}

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# OpenCV Video Capture
cap = cv2.VideoCapture(0)

print("Press 'P' to play the note for the current gesture. Press ESC to exit.")

current_label = None

def extract_landmarks(hand_landmarks):
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]
    return x_coords + y_coords

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    current_label = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            features = extract_landmarks(hand_landmarks)
            if len(features) == 42:
                feature_names = [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)]
                X_input_df = pd.DataFrame([features], columns=feature_names)
                prediction = model.predict(X_input_df)[0]
                current_label = label_encoder.inverse_transform([prediction])[0]

                # Display predicted label
                h, w, _ = frame.shape
                x = int(hand_landmarks.landmark[0].x * w)
                y = int(hand_landmarks.landmark[0].y * h)
                cv2.putText(frame, current_label, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the camera frame
    cv2.imshow("Real-Time Gesture to Note", frame)

    # Key listener
    key = cv2.waitKey(5) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('p') and current_label:
        note = gesture_to_note.get(current_label)
        if note and note in note_sounds:
            note_sounds[note].play()
            print(f"Played note: {note} for gesture: {current_label}")

cap.release()
cv2.destroyAllWindows()
