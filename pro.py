import cv2
import mediapipe as mp
import joblib
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,    
    max_num_hands=1,
    min_detection_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils
model = joblib.load('models/gesture_model.pkl')
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)
    panel = np.zeros((frame.shape[0],300,3),dtype=np.uint8)
    panel[:] = (64, 61, 61)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,hand_landmarks,mp_hands.HAND_CONNECTIONS
            )
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            
            coords = np.array(coords).reshape(1, -1)
            prediction = model.predict(coords)
            print(prediction[0])

            # Display on screen
            cv2.putText(panel, f"Gesture: {prediction[0]}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255 , 255, 255), 1)
 
    combined=np.hstack((frame,panel))
    cv2.imshow("Hand Landmark Test", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
