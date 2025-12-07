import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path


# Définition du chemin
Processed_path = Path("/home/ubuntu/examen-dvc/examen-dvc/data/processed_data")


# Lecture des dataframe train-test
X_train = pd.read_csv(Processed_path / "X_train.csv")
X_test = pd.read_csv(Processed_path / "X_test.csv")


# Conversion des nombres :
for col in X_train.columns:
    X_train[col] = (X_train[col].astype(str).str.replace(",", ".", regex=False).str.replace(" ", "", regex=False))
    X_test[col] = (X_test[col].astype(str).str.replace(",", ".", regex=False).str.replace(" ", "", regex=False))

    X_train[col] = pd.to_numeric(X_train[col], errors="ignore")
    X_test[col] = pd.to_numeric(X_test[col], errors="ignore")


# Transformation des colonnes de dates car elles génèrent des erreurs : scinder en année/mois/jour/heure
for col in X_train.columns:
    if X_train[col].dtype == "object":
        try:
            X_train[col] = pd.to_datetime(X_train[col])
            X_test[col] = pd.to_datetime(X_test[col])

            for df in [X_train, X_test]:
                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_day"] = df[col].dt.day
                df[f"{col}_hour"] = df[col].dt.hour
                df[f"{col}_dayofweek"] = df[col].dt.dayofweek
            # Suppression de la colonne originale
            X_train.drop(columns=[col], inplace=True)
            X_test.drop(columns=[col], inplace=True)

        except Exception:
            pass


# On sélectionne uniquement les colonnes numériques
X_train = X_train.select_dtypes(include=[np.number])
X_test = X_test.select_dtypes(include=[np.number])


# Mise à l'échelle
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Création des nouveaux datasets
pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(Processed_path / "X_train_scaled.csv", index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(Processed_path / "X_test_scaled.csv", index=False)


