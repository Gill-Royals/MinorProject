import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)


gesture_label = input("Enter Gesture Number (1-5): ")

file = open("gesture_data.csv", "a", newline="")
writer = csv.writer(file)

print("Press 's' to save sample")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:

            data = []

            for lm in hand.landmark:
                data.append(lm.x)
                data.append(lm.y)

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            key = cv2.waitKey(1)

            if key == ord('s'):
                data.append(gesture_label)
                writer.writerow(data)
                print("Sample Saved")

    cv2.imshow("Collect Data", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
file.close()