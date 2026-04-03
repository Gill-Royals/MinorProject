from datetime import datetime
from database import Database
import time

db=Database()
class SentenceFormer:
  def __init__(self):
    self.gesture_buffer=[]
    self.last_gesture= None
    self.last_time= None
    self.pause_duplicate= 3.0
    self.held_added= False

    

  def clear(self):
    self.gesture_buffer =[]
    self.last_gesture= None
    self.last_time=None
    self.held_added= False

  def add_gesture(self,gesture,confidence):

    timestamp=time.time()
    if gesture!=self.last_gesture:
    
      db.gesture( "test", gesture, confidence, timestamp)
      self.last_gesture = gesture

      self.last_time = timestamp
      self.held_added = False
      return True

    if gesture==self.last_gesture:
      if self.last_time:
        duration = (timestamp - self.last_time)
        if duration>=self.pause_duplicate and not self.held_added:
          db.gesture("test", gesture, confidence, timestamp)
          self.held_added = True
          return True
    return False


