import sqlite3
from datetime import datetime
class Database():
  def __init__(self):
    self.conn = sqlite3.connect("database.db")
    self.create()
  
  def create(self):
    self.conn.execute('''
      CREATE TABLE IF NOT EXISTS gestures(
      id INTEGER PRIMARY KEY,
      session_id TEXT,
      gesture CHAR(1),
      confidence REAL,
      timestamp DATETIME,
      is_sent BOOLEAN DEFAULT 0
      )
    ''')

    self.conn.execute('''
      CREATE TABLE IF NOT EXISTS sentences(
      id INTEGER PRIMARY KEY,
      session_id TEXT,
      raw_gestures TEXT,
      formed_sentence TEXT,
      created_at DATETIME
      )
    ''')
    self.conn.commit()

  def gesture(self,session_id,gesture,confidence, t):
    self.conn.execute("INSERT INTO gestures VALUES(NULL,?,?,?,?,0)",
    (session_id,gesture,confidence,datetime.now())
    )
    self.conn.commit()

  def sen(self, session_id,raw,formed):
    self.conn.execute("INSERT INTO sentences VALUES(NULL,?,?,?,?)",
    (session_id,raw,formed,datetime.now())
    )
    self.conn.commit()

  def show(self):
    cursor=self.conn.execute("SELECT* FROM gestures")
    for row in cursor:
      print(row)
  
  def clear(self):
    cursor=self.conn.execute("DELETE FROM gestures")
    self.conn.commit()

  def get_unsent(self):
    cursor=self.conn.cursor()
    cursor.execute('''
      SELECT id, gesture, timestamp FROM gestures WHERE is_sent =0 ORDER BY timestamp ASC
    ''')
    rows= cursor.fetchall()
    return rows

  def process_unsent(self):
      unsent = self.get_unsent()
    
      if unsent:
        gesture_sequence = ''.join([row[1] for row in unsent])
        return gesture_sequence
    
      return None
  
  def mark_sent(self):
    cursor=self.conn.cursor()
    cursor.execute('''
    UPDATE gestures SET is_sent=1 where is_sent=0
    ''')
    self.conn.commit()
  

 
#Database().show()
    