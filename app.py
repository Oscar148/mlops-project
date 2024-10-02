# app.py
import uvicorn
from fastapi import FastAPI
import pickle
import numpy as np
import pandas as pd

app = FastAPI()

# Cargar el modelo y el escalador
model = pickle.load(open('./models/ensemble_model.pkl', 'rb'))
scaler = pickle.load(open('./models/scaler.pkl', 'rb'))

@app.get("/")
def greet(name: str):
    return {
        "message": f"Hello, {name}!"
    }

@app.get("/health")
def health_check():
    return {
        "status": "OK"
    }

@app.post("/predict")
def predict(data: list[float]):
    # Crear un DataFrame a partir de los datos de entrada
    X = [{
        f"X{i+1}": x
        for i, x in enumerate(data)
    }]
    
    df = pd.DataFrame.from_records(X)

    # Escalar los datos
    df_scaled = scaler.transform(df)

    # Realizar la predicción
    prediction = model.predict(df_scaled)
    
    return {
        "prediction": int(prediction[0])
    }
    
if __name__ == "__main__":
    uvicorn.run("app:app", port=1234, reload=True)
