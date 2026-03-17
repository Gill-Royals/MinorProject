import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

df=pd.read_csv('data/hand_gesture_data.csv',header=None)

x=df.iloc[:,:-1]
y=df.iloc[:,-1]
print("Total Samples:",len(df))
# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(x , y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model
joblib.dump(model, 'models/gesture_model.pkl')
print("Model saved successfully!")
print(df['Gesture'].value_counts())