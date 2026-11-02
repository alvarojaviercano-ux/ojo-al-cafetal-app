# Paso 3 - Regresión logística: ¿la hoja está sana o enferma?

# 1. Importar librerías necesarias
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import caracteristicas as car

RAIZ = Path(__file__).resolve().parent
SALIDAS = RAIZ / "salidas"
MODELOS = RAIZ / "models"
SALIDAS.mkdir(exist_ok=True)
MODELOS.mkdir(exist_ok=True)

# 2. Carga de datos
data = pd.read_csv(RAIZ / "data.csv")
print(data.head())
print(data.info())

# 3. Definir variables
X = data[car.NOMBRES]      # las 20 características
y = data["target"]         # 0 = sana, 1 = enferma

# 4. Dividir en entrenamiento y prueba (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Escalar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Crear y entrenar el modelo
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

# 7. Predicciones
y_pred = logistic_model.predict(X_test)

# 8. Evaluación
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Sana", "Enferma"], yticklabels=["Sana", "Enferma"])
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de confusión")
plt.tight_layout()
plt.savefig(SALIDAS / "matriz_confusion_logistica.png", dpi=150)
plt.show()

print(classification_report(y_test, y_pred, target_names=["Sana", "Enferma"]))
accuracy = accuracy_score(y_test, y_pred)
print(f"Exactitud del modelo: {accuracy * 100:.2f}%")

# 9. Qué características pesan más
pesos = pd.Series(logistic_model.coef_[0], index=car.NOMBRES)
print("\nCaracterísticas que más empujan hacia 'enferma' (+) o 'sana' (-):")
print(pesos.sort_values(key=abs, ascending=False).head(8).round(2).to_string())

# 10. Guardar el modelo para la app de Render
joblib.dump({"scaler": scaler, "modelo": logistic_model, "columnas": car.NOMBRES},
            MODELOS / "logistica.joblib")
print("\nModelo guardado en models/logistica.joblib")
