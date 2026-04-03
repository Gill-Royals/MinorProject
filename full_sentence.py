import os
from groq import Groq

class SentenceBuilder:
    def __init__(self):
      self.client = Groq(api_key="Apikey here")
    
    def build_sentence(self, gesture_sequence):
        """
        Takes a sequence of gestures and returns a grammatically correct sentence
        Example input: "HLLOWORLD" -> "Hello world"
        """
        if not gesture_sequence:
            return None
        
        prompt = f"""You are a text correction assistant. Convert this gesture sequence into a proper English sentence:
        
Gesture sequence: {gesture_sequence}

Rules:
- Add spaces between words
- Fix spelling errors
- Make it grammatically correct
- Keep it concise
- Return ONLY the corrected sentence, nothing else
- Do not over translate just give barebone 
- Add question marks, commas etc

Corrected sentence:"""
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100
            )
            
            sentence = response.choices[0].message.content.strip()
            return sentence
            
        except Exception as e:
            print(f"Groq API Error: {e}")
            return None