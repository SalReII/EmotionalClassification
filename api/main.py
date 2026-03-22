from fastapi import FastAPI
from pydantic import BaseModel
from src.inference import predict

app = FastAPI()

class Request(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
def get_prediction(req: Request):
    star, probs = predict(req.text)
    return {
        "stars": star,
        "probabilities": probs
    }