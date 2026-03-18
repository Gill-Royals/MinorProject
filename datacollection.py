import cv2
import mediapipe as mp
import csv
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

# Create CSV file if not exists
file_name = "h_g_d.csv"
file_exists = os.path.isfile(file_name)

csv_file = open(file_name, mode="a", newline="")
csv_writer = csv.writer(csv_file)

print("Press keys 0–9 to collect gestures")
print("Press 'q' to quit")

sample_count = {str(i): 0 for i in range(10)}

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip & convert
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand
    result = hands.process(rgb)

    # Draw landmarks
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # Show instructions
    cv2.putText(frame, "Press 0-9 to collect | Q to quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

    # Show sample count
    y_offset = 60
    for key, count in sample_count.items():
        cv2.putText(frame, f"{key}: {count}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)
        y_offset += 20

    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    # If number key pressed (0–9)
    if key in [ord(str(i)) for i in range(10)]:
        label = chr(key)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                coords = []

                #  Use wrist as reference (relative coords)
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

            print(f"Saved gesture {label} | Total: {sample_count[label]}")
        else:
            print(" No hand detected!")

    # Quit
    elif key == ord('q'):
        break

# Cleanup
csv_file.close()
cap.release()
cv2.destroyAllWindows()

print("Data collection finished ")