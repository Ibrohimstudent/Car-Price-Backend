"""FastAPI backend. Ishga tushirish: uvicorn main:app --reload"""
import json

import pandas as pd
from catboost import CatBoostRegressor
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, create_model

app = FastAPI(title="Car Price Prediction API")

# --- CORS: frontend boshqa domenda turadi, shuning uchun ruxsat kerak ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://car-price-prediction-one-gilt.vercel.app/"],  # productionda frontend domeningizni yozing, masalan ["https://sizning-sayt.vercel.app"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model va metadata yuklash ---
model = CatBoostRegressor()
model.load_model("car_model.cbm")

with open("meta.json", encoding="utf-8") as f:
    meta = json.load(f)

with open("stats.json", encoding="utf-8") as f:
    stats = json.load(f)

FEATURES = meta["features"]

# --- So'rov (request) uchun schema avtomatik yasaymiz ---
fields = {}
for name in meta["cat"]:
    fields[name] = (str, ...)
for name, info in meta["num"].items():
    fields[name] = (int if info["is_int"] else float, ...)

CarInput = create_model("CarInput", **fields)


@app.get("/")
def root():
    return {"status": "ok", "message": "Car Price Prediction API ishlayapti"}


@app.get("/meta")
def get_meta():
    """Frontend forma qurish uchun kategoriyalar va min/max qiymatlarni oladi."""
    return meta


@app.get("/stats")
def get_stats():
    """Statistika sahifasi uchun model solishtiruvi va feature importance."""
    return stats


@app.post("/predict")
def predict(car: CarInput):
    row = pd.DataFrame([car.model_dump()])[FEATURES]
    price = float(model.predict(row)[0])
    return {"predicted_price": round(price, 2)}
