import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


# Définition des chemins
Raw_path = Path("/home/ubuntu/examen-dvc/examen-dvc/data/raw_data/raw.csv")
Processed_path = Path("/home/ubuntu/examen-dvc/examen-dvc/data/processed_data")
Processed_path.mkdir(parents=True, exist_ok=True)


# lecture du dataframe
df = pd.read_csv(Raw_path)


# Création dataset features / varaible cible
y = df.iloc[:, -1]
X = df.iloc[:, :-1]


# train-test plit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Création des datasets train-test
X_train.to_csv(Processed_path / "X_train.csv", index=False)
X_test.to_csv(Processed_path / "X_test.csv", index=False)
y_train.to_csv(Processed_path / "y_train.csv", index=False)
y_test.to_csv(Processed_path / "y_test.csv", index=False)