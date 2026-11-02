# Paso 4 - Comparar modelos con las mismas características

# 1. Importar librerías necesarias
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import caracteristicas as car

RAIZ = Path(__file__).resolve().parent

# 2. Carga de datos
data = pd.read_csv(RAIZ / "data.csv")
X = data[car.NOMBRES]
y = data["target"]

# 3. Modelos a comparar (los que necesitan escala la llevan dentro)
modelos = {
    "Regresión logística": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "KNN (k=7)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
    "Bosque aleatorio": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "Gradient boosting": HistGradientBoostingClassifier(random_state=42),
}

# 4. Validación cruzada de 5 partes
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("Exactitud (sana vs enferma), promedio de 5 partes:\n")
resultados = {}
for nombre, modelo in modelos.items():
    puntajes = cross_val_score(modelo, X, y, cv=cv, scoring="accuracy")
    resultados[nombre] = puntajes.mean()
    print(f"  {nombre:22s} {puntajes.mean() * 100:.2f}%  (±{puntajes.std() * 100:.2f})")

# 5. Acierto por clase con el mejor modelo
mejor = max(resultados, key=resultados.get)
print(f"\nMejor modelo: {mejor}")
y_pred = cross_val_predict(modelos[mejor], X, y, cv=cv)
data["acierto"] = (y_pred == y)
print("\nAcierto por clase:")
print((data.groupby("clase")["acierto"].agg(["mean", "count"])
       .rename(columns={"mean": "acierto", "count": "fotos"})
       .assign(acierto=lambda t: (t["acierto"] * 100).round(1)).to_string()))