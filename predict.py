import pandas as pd
import pickle

# Cargar el modelo entrenado
with open('./models/ensemble_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Cargar el dataset
data = pd.read_csv('./data/credit_pred.csv')

# Realizar predicciones (asumimos que todas las columnas son features)
predictions = model.predict(data)

# Añadir la columna 'Y' con las predicciones
data['Y'] = predictions

# Guardar el nuevo dataset con la columna 'Y'
data.to_csv('credit_pred_with_predictions.csv', index=False)

print("Predicciones guardadas en 'credit_pred_with_predictions.csv'")