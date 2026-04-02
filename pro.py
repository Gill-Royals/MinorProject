import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque

# Load trained model
model = joblib.load('Models/gesture_model.pkl')

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# Prediction smoothing
history = deque(maxlen=7)

# Confidence thresholds
LOW_CONF = 50

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip & convert
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand
    result = hands.process(rgb)

    # Side panel
    panel = np.zeros((frame.shape[0], 300, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # -------- FEATURE EXTRACTION (HAND-CENTERED) --------
            base_x = hand_landmarks.landmark[0].x
            base_y = hand_landmarks.landmark[0].y
            base_z = hand_landmarks.landmark[0].z

            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([
                    lm.x - base_x,
                    lm.y - base_y,
                    lm.z - base_z
                ])

            coords = np.array(coords).reshape(1, -1)

            # -------- PREDICTION --------
            prediction = model.predict(coords)[0]
            prob = model.predict_proba(coords)
            confidence = np.max(prob) * 100

            # -------- CONFIDENCE FILTER + SMOOTHING --------
            if confidence < LOW_CONF:
                final_prediction = "..."
            else:
                history.append(prediction)
                final_prediction = max(set(history), key=history.count)

            # Debug print
            print(f"{final_prediction} - {confidence:.2f}%")

            # -------- DISPLAY --------
            cv2.putText(panel, f"Gesture: {final_prediction}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2)

            cv2.putText(panel, f"Confidence: {confidence:.2f}%",
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2)

    else:
        cv2.putText(panel, "No Hand Detected",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2)

    # Combine webcam + panel
    combined = np.hstack((frame, panel))
    cv2.imshow("Hand Gesture Recognition", combined)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()