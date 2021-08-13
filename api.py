from fastapi import FastAPI
from inference import predict

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/predict")
async def fetch_predictions(text: str):
    prediction = predict(text)
    return {"Given Sentence" : text,
            "Sentiment for the given Sentence" : prediction['label'],
            "Negative" : prediction['negative'],
            "Positive" :  prediction['positive']}

# uvicorn api:app --reload
# http://127.0.0.1:8000/
# http://127.0.0.1:8000/docs  => swagger 
