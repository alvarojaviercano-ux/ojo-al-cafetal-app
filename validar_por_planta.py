# Paso 5 - Validar con plantas que el modelo nunca vio

# 1. Importar librerías necesarias
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score

import caracteristicas as car

RAIZ = Path(__file__).resolve().parent

# 2. Carga de datos
data = pd.read_csv(RAIZ / "data.csv")
X = data[car.NOMBRES]
y = data["target"]

# 3. Identificar la planta: del nombre C1P2E1.jpg se toma "C1P2"
data["planta"] = data["archivo"].str.extract(r"^(C\d+P\d+)")[0]
print(f"Plantas distintas: {data['planta'].nunique()}")
print("Fotos por planta (cuántas plantas tienen cada cantidad):")
print(data["planta"].value_counts().value_counts().sort_index().to_string())

# 4. Comparar: división normal vs. división por planta
modelo = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)

normal = cross_val_score(modelo, X, y, scoring="accuracy",
                         cv=StratifiedKFold(5, shuffle=True, random_state=42))
por_planta = cross_val_score(modelo, X, y, scoring="accuracy",
                             cv=GroupKFold(n_splits=5), groups=data["planta"])

print(f"\nDivisión normal:     {normal.mean() * 100:.2f}%")
print(f"División por planta: {por_planta.mean() * 100:.2f}%  (±{por_planta.std() * 100:.2f})")