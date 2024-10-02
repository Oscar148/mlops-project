# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
import pickle

def main():
    # Cargar el dataset
    data = pd.read_csv("./data/credit_train.csv")
    X = data.drop("Y", axis=1)
    Y = data["Y"]

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1234)

    # Escalar los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Definir los modelos
    xgb_model = xgb.XGBClassifier(
        scale_pos_weight=len(Y_train[Y_train == 0]) / len(Y_train[Y_train == 1]),  # Balanceo de clases
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.001,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=1234
    )

    lr_model = LogisticRegression(penalty='l2', C=0.1, solver='liblinear')
    rf_model = RandomForestClassifier(n_estimators=100, random_state=1234)

    # Crear un VotingClassifier con los modelos
    ensemble_model = VotingClassifier(estimators=[('xgb', xgb_model), ('lr', lr_model), ('rf', rf_model)], voting='soft')

    # Entrenar el modelo combinado
    ensemble_model.fit(X_train_scaled, Y_train)

    # Hacer predicciones
    Y_hat_train = ensemble_model.predict(X_train_scaled)
    Y_hat_test = ensemble_model.predict(X_test_scaled)

    # Evaluar el modelo
    f1_train = f1_score(Y_train, Y_hat_train)
    f1_test = f1_score(Y_test, Y_hat_test)
    
    accuracy_train = accuracy_score(Y_train, Y_hat_train)
    accuracy_test = accuracy_score(Y_test, Y_hat_test)

    print("F1 score train:", f1_train)
    print("F1 score test:", f1_test)
    print("Accuracy train:", accuracy_train)
    print("Accuracy test:", accuracy_test)

    # Guardar el modelo y el escalador
    with open("./models/ensemble_model.pkl", "wb") as file:
        pickle.dump(ensemble_model, file)

    with open("./models/scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)

if __name__ == '__main__':
    main()
