# Paso 7 - KNN: ¿qué enfermedad tiene la hoja?

# 1. Importar librerías necesarias
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import caracteristicas as car

RAIZ = Path(__file__).resolve().parent
SALIDAS = RAIZ / "salidas"
MODELOS = RAIZ / "models"
SALIDAS.mkdir(exist_ok=True)
MODELOS.mkdir(exist_ok=True)

# 2. Carga de datos: solo hojas enfermas
data = pd.read_csv(RAIZ / "data.csv")
data = data[data["target"] == 1].reset_index(drop=True)
data["planta"] = data["archivo"].str.extract(r"^(C\d+P\d+)")[0]
print(data["clase"].value_counts().to_string())

# 3. Definir variables
X = data[car.NOMBRES]
y = data["clase"]

# 4. Dividir 80% / 20% por planta
division = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
indices_train, indices_test = next(division.split(X, y, groups=data["planta"]))
X_train, X_test = X.iloc[indices_train], X.iloc[indices_test]
y_train, y_test = y.iloc[indices_train], y.iloc[indices_test]

# 5. Escalar los datos (KNN usa distancias, así que es obligatorio)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Elegir k: se prueba con validación cruzada solo en el entrenamiento
grupos_train = data["planta"].iloc[indices_train]
print("\nExactitud según k (validación cruzada por planta):")
mejor_k, mejor_acc = 1, 0
for k in [1, 3, 5, 7, 9, 11, 15, 21]:
    acc = cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train, y_train,
                          cv=GroupKFold(n_splits=5), groups=grupos_train).mean()
    print(f"  k = {k:2d}: {acc * 100:.2f}%")
    if acc > mejor_acc:
        mejor_k, mejor_acc = k, acc
print(f"Mejor k: {mejor_k}")

# 7. Crear y entrenar el modelo
knn_model = KNeighborsClassifier(n_neighbors=mejor_k)
knn_model.fit(X_train, y_train)

# 8. Predicciones
y_pred = knn_model.predict(X_test)

# 9. Evaluación
etiquetas = sorted(y.unique())
conf_matrix = confusion_matrix(y_test, y_pred, labels=etiquetas)
plt.figure(figsize=(7, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=etiquetas, yticklabels=etiquetas)
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de confusión - KNN")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(SALIDAS / "matriz_confusion_knn.png", dpi=150)
plt.show()

print(classification_report(y_test, y_pred, zero_division=0))
accuracy = accuracy_score(y_test, y_pred)
print(f"Exactitud del modelo: {accuracy * 100:.2f}%")

# 10. Guardar el modelo para la app de Render
joblib.dump({"scaler": scaler, "modelo": knn_model, "columnas": car.NOMBRES},
            MODELOS / "knn.joblib")
print("\nModelo guardado en models/knn.joblib")