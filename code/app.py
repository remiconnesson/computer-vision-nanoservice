from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from fastai.vision.all import load_learner


class Task(BaseModel):
    img_path: str

app = FastAPI()

ai = { "model" : False }

@app.on_event("startup")
async def startup_event():
    print("loading model...")
    ai["model"] = load_learner("model.pkl")
    print("loading model ✅")

@app.post("/inference")
async def predict(task: Task):
    img_filename = Path(task.img_path).name
    prediction = ai["model"].predict(Path("data") / img_filename )
    return prediction[0]

@app.post("/ping")
async def health_check():
    img_path = "birdy.jpg"
    prediction = ai["model"].predict(img_path)
    return prediction[0]


