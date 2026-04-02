import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque

# LOCK CONCEPT

# ==============================
# CONFIG
# ==============================
CONF_THRESHOLD = 40
HISTORY_SIZE = 7

# ==============================
# MediaPipe Init
# ==============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# ==============================
# Load Model
# ==============================
model = joblib.load("Models/gesture_model_letters.pkl")

# ==============================
# Load Gesture Chart
# ==============================
gesture_chart = cv2.imread("letters.jpeg")
if gesture_chart is None:
    print("❌ letters.jpeg not found!")

# ==============================
# Webcam
# ==============================
cap = cv2.VideoCapture(0)

# ==============================
# Variables
# ==============================
history = deque(maxlen=HISTORY_SIZE)
sentence = ""
locked = False   # 🔒 LOCK FLAG

# ==============================
# FULLSCREEN WINDOW
# ==============================
cv2.namedWindow("Gesture AI System", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "Gesture AI System",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

# ==============================
# MAIN LOOP
# ==============================
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    final_pred = "..."
    confidence = 0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Feature extraction (hand-centered)
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

            # Smoothing
            if confidence > CONF_THRESHOLD:
                history.append(prediction)

                if len(history) == HISTORY_SIZE:
                    final_pred = max(set(history), key=history.count)
                else:
                    final_pred = "..."
            else:
                final_pred = "..."

            # 🔒 LOCK CONCEPT (CORE LOGIC)
            if final_pred != "..." and not locked:
                sentence += str(final_pred)
                locked = True   # lock after one input

    else:
        final_pred = "No Hand"
        history.clear()
        locked = False   # 🔓 UNLOCK when hand removed

    # ==============================
    # KEYBOARD CONTROLS
    # ==============================
    key = cv2.waitKey(1) & 0xFF

    if key == 32:  # SPACE
        sentence += " "
    elif key == 8:  # BACKSPACE
        sentence = sentence[:-1]
    elif key == ord('c'):  # CLEAR
        sentence = ""
    elif key == ord('q'):
        break

    # ==============================
    # UI CANVAS (60-40 SPLIT)
    # ==============================
    canvas_width = 1400
    canvas_height = 800

    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 20)

    # Split
    right_width = int(canvas_width * 0.4)
    left_width = canvas_width - right_width

    # ===== RIGHT SIDE IMAGE =====
    if gesture_chart is not None:
        img_resized = cv2.resize(gesture_chart, (right_width, canvas_height))
        canvas[:, left_width:canvas_width] = img_resized

    # ===== CAMERA =====
    cam = cv2.resize(frame, (left_width - 40, 380))
    canvas[20:400, 20:left_width - 20] = cam

    # ===== BOTTOM PANEL =====
    cv2.rectangle(canvas, (20, 420), (left_width - 20, 780), (40, 40, 40), -1)

    # Letter
    cv2.putText(canvas, f"Letter: {final_pred}",
                (40, 470),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2)

    # Confidence
    cv2.putText(canvas, f"Confidence: {confidence:.1f}%",
                (40, 520),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2)

    # Output
    cv2.putText(canvas, "Output:",
                (40, 570),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (180, 180, 180),
                1)

    cv2.putText(canvas, sentence[-35:],
                (40, 630),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2)

    # Controls
    cv2.putText(canvas, "SPACE = Space | BACKSPACE = Delete | C = Clear | Q = Quit",
                (40, 740),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (150, 150, 150),
                1)

    # Show
    cv2.imshow("Gesture AI System", canvas)

cap.release()
cv2.destroyAllWindows()