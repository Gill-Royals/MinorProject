import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# Load model
model = joblib.load("gesture_model_letters.pkl")

# Webcam
cap = cv2.VideoCapture(0)

# Smooth prediction
history = deque(maxlen=5)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    # Side panel
    panel = np.zeros((frame.shape[0], 300, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Same feature logic
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

            prediction = model.predict(coords)[0]
            prob = model.predict_proba(coords)
            confidence = np.max(prob) * 100

            # Smooth output
            history.append(prediction)
            final_pred = max(set(history), key=history.count)

            print(f"{final_pred} - {confidence:.2f}%")

            # Display
            cv2.putText(panel, f"Letter: {final_pred}",
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

    combined = np.hstack((frame, panel))
    cv2.imshow("A-Z Gesture Recognition", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()