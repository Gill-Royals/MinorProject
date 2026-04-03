import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque
import time
from database import Database
from sentenceformer import SentenceFormer
from full_sentence import SentenceBuilder

class Mediapipe:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils


# Load trained model
class ModelLoader:
    def __init__(self):
        model_num='Models/gesture_model.pkl'
        model_let='Models/gesture_model_letters.pkl'

        self.model_letters = joblib.load(model_let)
        self.model_numbers = joblib.load(model_num)

        self.active = self.model_letters
        self.mode = 'LETTERS'

        self.flag = False
    
    def Check(self, final_prediction):
        current_time= time.time()
        if final_prediction == 'Thumbs Up' and not self.flag:
            self.flag=True
            if self.active== self.model_numbers:
                self.active = self.model_letters
                self.mode = 'LETTERS'
            elif self.active == self.model_letters:
                self.active = self.model_numbers
                self.mode = 'NUMBERS'
        elif final_prediction!='Thumbs Up':
            self.flag=False      
        return self.mode
            
            

# Webcam
class Webcam:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
    
    def frame(self):
        success,frame= self.cap.read()
        if not success:
            print("Error opening webcam!")
            exit()
        else:
            return frame
    
    def release(self):
        self.cap.release()
        


#  Prediction smoothing (reduces flicker)
class Prediction:
    def __init__(self):
        self.history = deque(maxlen=5)

    def calculation(self,coords,model):
            self.prediction = model.predict(coords)[0]
            prob = model.predict_proba(coords)
            confidence = np.max(prob) * 100
            return self.prediction,confidence

    def smooth(self, prediction):
        self.history.append(prediction)
        final_prediction = max(set(self.history), key=self.history.count)
        return final_prediction

class window:
    def __init__(self):
        self.panel = np.zeros((480, 500, 3), dtype=np.uint8)
        self.panel[:] = (40, 40, 40)

    def handDetected(self,frame,final_prediction,confidence,mode):
        self.panel[:] = (40, 40, 40)
        cv2.putText(self.panel, f"Gesture: {final_prediction}",(10, 60),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255, 255),2)

        cv2.putText(self.panel, f"Confidence: {confidence:.2f}%",(10, 120),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255, 255, 255),2)
        
        cv2.putText(self.panel,f"Mode: {mode}",(10,180),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255, 255, 255),2)
    def switching(self):
        self.panel[:] = (40,40,40)
        cv2.putText(self.panel, f"Switching Modes....",(10, 60),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255, 255, 255),2)
                       

    def nohand(self, last_sentence):
        self.panel[:] = (40, 40, 40)
        cv2.putText(self.panel, f"Translation: {last_sentence} ",(10, 80),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255, 255, 255),2)

    def show(self, frame):
        combined = np.hstack((frame, self.panel))
        cv2.imshow("Hand Gesture Recognition", combined)

class landmarks:
    def __init__(self,hand_landmarks,mp_draw,mp_hands,frame):
        mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
        )
    def logic(self,hand_landmarks):
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
        return coords


mp_object = Mediapipe()
model_loader = ModelLoader()
cam = Webcam()
hands = mp_object.hands
mp_draw = mp_object.mp_draw
mp_hands = mp_object.mp_hands
smooth = Prediction()
window = window()
db= Database()
fsb=SentenceBuilder()
raw=SentenceFormer()

framecount=0
final_prediction=""
confidence=0
no_hand_threshold=30
no_hand_counter =0
last_sentence=""

while True:

    frame = cam.frame()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            
            landmark = landmarks(hand_landmarks,mp_draw,mp_hands,frame)

            coords = landmark.logic(hand_landmarks)

            framecount+=1
            
            if framecount%2==0:

                prediction, confidence = smooth.calculation(coords,model_loader.active)

                final_prediction= smooth.smooth(prediction)
                if final_prediction!='Thumbs Up':
                    raw.add_gesture(final_prediction,confidence)

            model_loader.Check(final_prediction)

            if final_prediction!='Thumbs Up':
                window.handDetected(frame,final_prediction,confidence,model_loader.mode)
                #print(f"{final_prediction} - {confidence:.2f}%")
            else:
                window.switching()

    else:
        no_hand_counter += 1
        
        if no_hand_counter >= no_hand_threshold:
            raw_gestures = db.process_unsent()
            
            if raw_gestures:
                full = fsb.build_sentence(raw_gestures)
                
                if full:
                    db.mark_sent()
                    db.sen(session_id="default", raw=raw_gestures, formed=full)
                    last_sentence=full
                    print(f"Formed sentence: {full}")
            
            no_hand_counter = 0  # Reset after processing

        window.nohand(last_sentence)


    window.show(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()