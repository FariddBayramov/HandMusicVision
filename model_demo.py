import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load model and label encoder
model = joblib.load("models/gesture_classifier.pkl")
label_encoder = joblib.load("models/gesture_labels.pkl")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# OpenCV Video Capture
cap = cv2.VideoCapture(0)

def extract_landmarks(hand_landmarks):
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]
    return x_coords + y_coords

print("Starting real-time prediction. Press ESC to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            features = extract_landmarks(hand_landmarks)
            if len(features) == 42:  # 21 x and 21 y
                X_input = np.array(features).reshape(1, -1)
                prediction = model.predict(X_input)[0]
                label = label_encoder.inverse_transform([prediction])[0]

                # Display prediction on frame
                h, w, _ = frame.shape
                x = int(hand_landmarks.landmark[0].x * w)
                y = int(hand_landmarks.landmark[0].y * h)
                cv2.putText(frame, label, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Real-Time Gesture Recognition", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
