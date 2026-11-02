# Paso 6 - Modelo final: bosque aleatorio (sana o enferma)

# 1. Importar librerías necesarias
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

import caracteristicas as car

RAIZ = Path(__file__).resolve().parent
SALIDAS = RAIZ / "salidas"
MODELOS = RAIZ / "models"
SALIDAS.mkdir(exist_ok=True)
MODELOS.mkdir(exist_ok=True)

# 2. Carga de datos
data = pd.read_csv(RAIZ / "data.csv")
data["planta"] = data["archivo"].str.extract(r"^(C\d+P\d+)")[0]

# 3. Definir variables
X = data[car.NOMBRES]
y = data["target"]          # 0 = sana, 1 = enferma

# 4. Dividir 80% / 20% por planta (una planta nunca queda en los dos lados)
division = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
indices_train, indices_test = next(division.split(X, y, groups=data["planta"]))
X_train, X_test = X.iloc[indices_train], X.iloc[indices_test]
y_train, y_test = y.iloc[indices_train], y.iloc[indices_test]
print(f"Entrenamiento: {len(X_train)} fotos | Prueba: {len(X_test)} fotos")

# 5. Crear y entrenar el modelo (el bosque no necesita escalar los datos)
forest_model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
forest_model.fit(X_train, y_train)

# 6. Predicciones
y_pred = forest_model.predict(X_test)

# 7. Evaluación
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Sana", "Enferma"], yticklabels=["Sana", "Enferma"])
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de confusión - Bosque aleatorio")
plt.tight_layout()
plt.savefig(SALIDAS / "matriz_confusion_bosque.png", dpi=150)
plt.show()

print(classification_report(y_test, y_pred, target_names=["Sana", "Enferma"]))
accuracy = accuracy_score(y_test, y_pred)
print(f"Exactitud del modelo: {accuracy * 100:.2f}%")

# 8. Acierto por clase en la prueba
prueba = data.iloc[indices_test].copy()
prueba["acierto"] = (y_pred == y_test.values)
print("\nAcierto por clase (prueba):")
print((prueba.groupby("clase")["acierto"].mean() * 100).round(1).to_string())

# 9. Características más importantes
importancias = pd.Series(forest_model.feature_importances_, index=car.NOMBRES)
print("\nCaracterísticas más importantes:")
print(importancias.sort_values(ascending=False).head(8).round(3).to_string())

# 10. Guardar el modelo para la app de Render
joblib.dump({"modelo": forest_model, "columnas": car.NOMBRES},
            MODELOS / "bosque.joblib")
print("\nModelo guardado en models/bosque.joblib")