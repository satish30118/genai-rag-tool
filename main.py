from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from model import Recommender
import os

app = FastAPI()

# ✅ Serve static frontend files from the 'frontend' folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# ✅ CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In prod, replace with frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load model
recommender = Recommender("SHL_dataset.csv")

# ✅ Schema
class RequestModel(BaseModel):
    text: str

# ✅ API root
@app.get("/")
def read_root():
    return FileResponse("static/index.html")  # 👈 serves the index.html page

# ✅ POST endpoint for recommendation
@app.post("/recommend")
def get_recommendation(request: RequestModel):
    results = recommender.recommend(request.text)
    return {"recommendations": results}
