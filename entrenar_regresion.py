# Paso 12 - Regresión lineal: ¿qué porcentaje de la hoja está afectado?
# El dataset no trae el porcentaje real, solo el nivel de roya (método OIRSA).
# Se usa el punto medio del rango de cada nivel como objetivo:
#   sana 0% | nivel 1 (1-5%) -> 3 | nivel 2 (6-20%) -> 13 | nivel 3 (21-50%) -> 35.5 | nivel 4 (>50%) -> 75

# 1. Importar librerías necesarias
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

import caracteristicas as car

RAIZ = Path(__file__).resolve().parent
SALIDAS = RAIZ / "salidas"
MODELOS = RAIZ / "models"
SALIDAS.mkdir(exist_ok=True)
MODELOS.mkdir(exist_ok=True)

PORCENTAJE_POR_NIVEL = {"healthy": 0, "rust_level_1": 3, "rust_level_2": 13,
                        "rust_level_3": 35.5, "rust_level_4": 75}

# 2. Carga de datos: características con GrabCut (como en la app) + etiquetas
etiquetas = pd.read_csv(RAIZ / "data.csv")[["archivo", "clase"]]
data = pd.read_csv(RAIZ / "data_grabcut.csv").merge(etiquetas, on="archivo")
data = data[data["clase"].isin(PORCENTAJE_POR_NIVEL)].reset_index(drop=True)  # sin ácaro rojo
data["porcentaje"] = data["clase"].map(PORCENTAJE_POR_NIVEL)
data["planta"] = data["archivo"].str.extract(r"^(C\d+P\d+)")[0]
print(f"Fotos: {len(data)}")

# 3. Definir variables
X = data[car.NOMBRES]
y = data["porcentaje"]

# 4. Dividir 80% / 20% por planta
division = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
indices_train, indices_test = next(division.split(X, y, groups=data["planta"]))
X_train, X_test = X.iloc[indices_train], X.iloc[indices_test]
y_train, y_test = y.iloc[indices_train], y.iloc[indices_test]
print(f"Entrenamiento: {len(X_train)} fotos | Prueba: {len(X_test)} fotos")

# 5. Escalar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Crear y entrenar el modelo
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# 7. Predicciones (un porcentaje no puede ser menor que 0 ni mayor que 100)
y_pred = np.clip(linear_model.predict(X_test), 0, 100)

# 8. Evaluación
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"\nError absoluto medio (MAE): {mae:.1f} puntos de porcentaje")
print(f"Raíz del error cuadrático (RMSE): {rmse:.1f}")
print(f"R²: {r2:.2f}")

plt.figure(figsize=(6, 5))
plt.scatter(y_test + np.random.RandomState(0).uniform(-1.5, 1.5, len(y_test)), y_pred,
            alpha=0.4, color="#A5531A")
plt.plot([0, 100], [0, 100], color="#2F6B47", linestyle="--", label="predicción perfecta")
plt.xlabel("Porcentaje de referencia (punto medio del nivel)")
plt.ylabel("Porcentaje predicho")
plt.title("Regresión lineal: real vs. predicho")
plt.legend()
plt.tight_layout()
plt.savefig(SALIDAS / "regresion_real_vs_predicho.png", dpi=150)
plt.show()

# 9. Predicción promedio por nivel y acierto de nivel
prueba = data.iloc[indices_test].copy()
prueba["predicho"] = y_pred
print("\nPorcentaje predicho promedio por nivel (prueba):")
print(prueba.groupby("clase")["predicho"].agg(["mean", "std", "count"]).round(1).to_string())

# Convertir el porcentaje predicho en el nivel más cercano (cortes entre puntos medios)
cortes = [1.5, 8, 24.25, 55.25]
niveles = ["healthy", "rust_level_1", "rust_level_2", "rust_level_3", "rust_level_4"]
prueba["nivel_predicho"] = [niveles[np.searchsorted(cortes, p)] for p in y_pred]
exacto = (prueba["nivel_predicho"] == prueba["clase"]).mean()
cerca = (abs(prueba["nivel_predicho"].map(niveles.index) - prueba["clase"].map(niveles.index)) <= 1).mean()
print(f"\nNivel correcto: {exacto * 100:.1f}% | Nivel correcto o vecino: {cerca * 100:.1f}%")

# 10. Características con más peso
pesos = pd.Series(linear_model.coef_, index=car.NOMBRES)
print("\nCaracterísticas con más peso:")
print(pesos.sort_values(key=abs, ascending=False).head(6).round(1).to_string())

# 11. Guardar el modelo
joblib.dump({"scaler": scaler, "modelo": linear_model, "columnas": car.NOMBRES},
            MODELOS / "regresion.joblib")
print("\nModelo guardado en models/regresion.joblib")