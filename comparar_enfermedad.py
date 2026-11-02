# Paso 8 - ¿Qué enfermedad tiene la hoja? Comparar modelos

# 1. Importar librerías necesarias
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import caracteristicas as car

RAIZ = Path(__file__).resolve().parent

# 2. Carga de datos: solo hojas enfermas
data = pd.read_csv(RAIZ / "data.csv")
data = data[data["target"] == 1].reset_index(drop=True)
data["planta"] = data["archivo"].str.extract(r"^(C\d+P\d+)")[0]
X = data[car.NOMBRES]

# 3. Dos versiones de las etiquetas
etiquetas = {
    "5 clases": data["clase"],
    "4 clases (roya 3 y 4 unidas)": data["clase"].replace(
        {"rust_level_3": "rust_level_3_4", "rust_level_4": "rust_level_3_4"}),
}

# 4. Modelos a comparar
modelos = {
    "KNN (k=9)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=9)),
    "Bosque aleatorio": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1),
    "Gradient boosting": HistGradientBoostingClassifier(
        class_weight="balanced", random_state=42),
}

# 5. Validación cruzada por planta
# Se muestran dos números: exactitud normal y exactitud promedio por clase
# (la segunda no se deja engañar por las clases grandes)
cv = GroupKFold(n_splits=5)
for nombre_et, y in etiquetas.items():
    print(f"\n=== {nombre_et} ===")
    for nombre, modelo in modelos.items():
        y_pred = cross_val_predict(modelo, X, y, cv=cv, groups=data["planta"])
        acc = (y_pred == y).mean()
        bal = balanced_accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average="macro")
        print(f"  {nombre:20s} exactitud {acc * 100:5.1f}% | "
              f"promedio por clase {bal * 100:5.1f}% | F1 macro {f1:.2f}")

# 6. Acierto por clase con el bosque aleatorio (5 clases)
y = etiquetas["5 clases"]
y_pred = cross_val_predict(modelos["Bosque aleatorio"], X, y, cv=cv,
                           groups=data["planta"])
data["acierto"] = (y_pred == y)
print("\nAcierto por clase con el bosque aleatorio (5 clases):")
print((data.groupby("clase")["acierto"].mean() * 100).round(1).to_string())