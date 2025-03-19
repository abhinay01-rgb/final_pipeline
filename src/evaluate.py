import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/features.csv")
X = df.drop(columns=["target"])
y = df["target"]

model = joblib.load("models/model.pkl")
predictions = model.predict(X)

accuracy = accuracy_score(y, predictions)
with open("metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}\n")

print("Model evaluation complete.")
