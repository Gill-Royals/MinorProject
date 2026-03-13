import cv2
import mediapipe as mp
import pickle
import numpy as np
from collections import deque

# Load trained model
model = pickle.load(open("gesture_model.pkl", "rb"))

# Gesture name mapping
gesture_names = {
    1: "One ",
    2: "Peace ",
    3: "Three ",
    4: "Four ",
    5: "Five "
}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Prediction smoothing
pred_queue = deque(maxlen=10)

while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for hand in result.multi_hand_landmarks:

            data = []

            for lm in hand.landmark:
                data.append(lm.x)
                data.append(lm.y)

            data = np.array(data).reshape(1, -1)

            # Prediction
            pred = int(model.predict(data)[0])

            # Confidence
            probs = model.predict_proba(data)
            confidence = np.max(probs) * 100

            pred_queue.append(pred)
            final_pred = max(set(pred_queue), key=pred_queue.count)

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # Gesture name
            gesture_text = gesture_names.get(final_pred, "Unknown")

            # Display gesture
            cv2.putText(frame,
                        f"Gesture: {gesture_text}",
                        (40, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2)

            # Display confidence
            cv2.putText(frame,
                        f"Confidence: {confidence:.2f}%",
                        (40, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        2)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()