# !pip install fastapi uvicorn

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Se inicializa FastAPI
app = FastAPI()

# Estructura de entrada
class WaterSample(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

# Se carga el modelo optimizado
def load_model():
    with open("models/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# Ruta GET para describir el modelo
@app.get("/")
def read_root():
    return {
        "description": "Modelo de clasificación de potabilidad del agua",
        "problem": "Predecir si el agua es potable basándose en varias características fisicoquímicas.",
        "input": {
            "ph": "Valor del pH del agua.",
            "Hardness": "Dureza del agua.",
            "Solids": "Cantidad total de sólidos disueltos.",
            "Chloramines": "Concentración de cloraminas.",
            "Sulfate": "Concentración de sulfato.",
            "Conductivity": "Conductividad del agua.",
            "Organic_carbon": "Cantidad de carbono orgánico.",
            "Trihalomethanes": "Concentración de trihalometanos.",
            "Turbidity": "Turbidez del agua."
        },
        "output": "1 si el agua es potable, 0 si no lo es."
    }

# Ruta POST para predecir la potabilidad del agua
@app.post("/potabilidad/")
def predict_potability(sample: WaterSample):
    sample_data = np.array([[sample.ph, sample.Hardness, sample.Solids, sample.Chloramines, 
                             sample.Sulfate, sample.Conductivity, sample.Organic_carbon, 
                             sample.Trihalomethanes, sample.Turbidity]])
    
    # Predicción
    prediction = model.predict(sample_data)
    
    return {"potabilidad": int(prediction[0])}