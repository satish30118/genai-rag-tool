from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import Recommender

app = FastAPI()

# ✅ Enable CORS for frontend access (you can restrict origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 👈 Replace "*" with actual domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load recommender model with dataset
recommender = Recommender("SHL_dataset.csv")

# ✅ Request schema for POST request body
class RequestModel(BaseModel):
    text: str

# ✅ Root endpoint to test if API is working
@app.get("/")
def read_root():
    return {"message": "API is working"}

# ✅ Recommendation endpoint
@app.post("/recommend")
def get_recommendation(request: RequestModel):
    results = recommender.recommend(request.text)
    return {"recommendations": results}
