import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

loaded_clf = joblib.load('sentiment_analysis_model.joblib')
target = {0: 'Positive', 2: 'Negative', 1: 'Neutral'}

app = FastAPI()

class TextInput(BaseModel):
    text: str

def predict_sentiment(text):
    if isinstance(text, str):
        text = [text]
    prediction = loaded_clf.predict(text)
    return target[prediction[0]]

@app.post("/predict")
def predict(input: TextInput):
    sentiment = predict_sentiment(input.text)
    return {"sentiment": sentiment}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
