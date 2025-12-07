import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import joblib
import json



# Définition des chemins
Processed_path = Path("/home/ubuntu/examen-dvc/examen-dvc/data/processed_data")
Model_path = Path("/home/ubuntu/examen-dvc/examen-dvc/models/models/")
Metrics_path = Path("/home/ubuntu/examen-dvc/examen-dvc/metrics/")
Metrics_path.mkdir(parents=True, exist_ok=True)


# Lecture des dataframes test
X_test = pd.read_csv(Processed_path / "X_test_scaled.csv")
y_test = pd.read_csv(Processed_path / "y_test.csv")


# Chargement du modèle
model = joblib.load(Model_path / "trained_model.pkl")


# Prediction
y_pred = model.predict(X_test)


# Indicateurs
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# Enregistrement des scores en json
scores = {"mse": mse, "r2": r2}

with open(Metrics_path / "scores.json", "w") as f:
    json.dump(scores, f, indent=4)


# Création dataset des prédictions
pred_df = pd.DataFrame({"y_true": y_test.values.ravel(), "y_pred": y_pred})


# Export en csv
pred_df.to_csv(Processed_path / "predictions.csv", index=False)

