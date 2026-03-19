import cv2
import mediapipe as mp
import csv
import os

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# CSV file
file_name = "Data/h_g_d_letters.csv"
csv_file = open(file_name, mode="a", newline="")
csv_writer = csv.writer(csv_file)

print("Press A–Z to collect data")
print("Press Q to quit")

# Count samples
sample_count = {chr(i): 0 for i in range(ord('A'), ord('Z')+1)}

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    # Draw landmarks
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display info
    cv2.putText(frame, "Press A-Z to collect | Q to quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

    y_offset = 60
    for k, v in sample_count.items():
        cv2.putText(frame, f"{k}: {v}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        y_offset += 15

    cv2.imshow("Letter Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    # A-Z detection
    if (ord('A') <= key <= ord('Z')) or (ord('a') <= key <= ord('z')):
        label = chr(key).upper()

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                coords = []

                # Relative coordinates
                base_x = hand_landmarks.landmark[0].x
                base_y = hand_landmarks.landmark[0].y
                base_z = hand_landmarks.landmark[0].z

                for lm in hand_landmarks.landmark:
                    coords.extend([
                        lm.x - base_x,
                        lm.y - base_y,
                        lm.z - base_z
                    ])

                coords.append(label)
                csv_writer.writerow(coords)

                sample_count[label] += 1

            print(f"Saved {label} | Count: {sample_count[label]}")
        else:
            print("No hand detected")

    elif key == ord('q'):
        break

csv_file.close()
cap.release()
cv2.destroyAllWindows()

print("✅ Letter data collection completed")