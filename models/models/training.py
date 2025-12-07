import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
import joblib


# Définition des chemins
Processed_path = Path("/home/ubuntu/examen-dvc/examen-dvc/data/processed_data")
Model_path = Path("/home/ubuntu/examen-dvc/examen-dvc/models/models/")
Model_path.mkdir(parents=True, exist_ok=True)


# Lecture des dataframes train
X_train = pd.read_csv(Processed_path / "X_train_scaled.csv")
y_train = pd.read_csv(Processed_path / "y_train.csv")


# pour avoir qu'une seule dimension pour y_train :
y_train = y_train.values.ravel()


# Chargement des best params 
best_params = joblib.load(Model_path / "best_params.pkl")


# Entrainement du modèle avec les best params
model = RandomForestRegressor(**best_params, random_state=42)
model.fit(X_train, y_train)


# Sauvegarde du modèle entraîné
joblib.dump(model, Model_path / "trained_model.pkl")




