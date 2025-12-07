import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
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


# Modèle : RF
model = RandomForestRegressor(random_state=42)


# Dico des paramètres
param_grid = {
"n_estimators": [100, 200],
"max_depth": [None, 10, 20],
"min_samples_split": [2, 5],
}


# GridSearch
grid = GridSearchCV(model, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)

grid.fit(X_train, y_train)


# Sauvegarde des best params
joblib.dump(grid.best_params_, Model_path / "best_params.pkl")