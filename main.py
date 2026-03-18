import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# 🔹 Mode selection using input (FIXED)
print("Press N for Numbers (0–9)")
print("Press A for Alphabets (A–Z)")

while True:
    choice = input("Enter N or A: ").lower()

    if choice == 'n':
        mode = "numbers"
        model = joblib.load("gesture_model.pkl")
        print(" Numbers mode selected")
        break

    elif choice == 'a':
        mode = "alphabets"
        model = joblib.load("gesture_model_letters.pkl")
        print(" Alphabets mode selected")
        break

    else:
        print(" Invalid input, try again")

# Webcam
cap = cv2.VideoCapture(0)

# Prediction smoothing
history = deque(maxlen=5)

while True:
    success, frame = cap.read()
    if not success:
        print(" Camera not working")
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

            # Relative coordinates
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

            # Prediction
            prediction = model.predict(coords)[0]
            prob = model.predict_proba(coords)
            confidence = np.max(prob) * 100

            # Smooth prediction
            history.append(prediction)
            final_pred = max(set(history), key=history.count)

            # Console output
            print(f"{final_pred} - {confidence:.2f}%")

            # Mode display
            cv2.putText(panel, f"Mode: {mode.upper()}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2)

            # Label display
            label_text = "Number" if mode == "numbers" else "Letter"

            cv2.putText(panel, f"{label_text}: {final_pred}",
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2)

            cv2.putText(panel, f"Confidence: {confidence:.2f}%",
                        (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2)

    else:
        cv2.putText(panel, "No Hand Detected",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2)

    # Combine frame + panel
    combined = np.hstack((frame, panel))
    cv2.imshow("Gesture Recognition System", combined)

    # Controls
    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord('q'):
        break

    #  Switch modes LIVE (BONUS )
    elif key == ord('n'):
        mode = "numbers"
        model = joblib.load("gesture_model.pkl")
        history.clear()
        print(" Switched to Numbers")

    elif key == ord('a'):
        mode = "alphabets"
        model = joblib.load("gesture_model_letters.pkl")
        history.clear()
        print(" Switched to Alphabets")

# Cleanup
cap.release()
cv2.destroyAllWindows()