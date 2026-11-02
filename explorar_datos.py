# Paso 1 - Explorar el dataset RoCoLe

# 1. Importar librerías necesarias
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# 2. Carga de datos
RAIZ = Path(__file__).resolve().parent
DATOS = RAIZ / "data" / "RoCoLe A robusta coffee leaf images dataset"
FOTOS = DATOS / "Photos"
EXCEL = DATOS / "Annotations" / "RoCoLe-classes.xlsx"
SALIDAS = RAIZ / "salidas"
SALIDAS.mkdir(exist_ok=True)

# Cargar las etiquetas desde el archivo de Excel
data = pd.read_excel(EXCEL)
data = data.rename(columns={"File": "archivo",
                            "Binary.Label": "binaria",
                            "Multiclass.Label": "clase"})

# Explorar el conjunto de datos
print(data.head())
print(data.info())
print(data.describe())

# 3. Conteo de clases
conteo = data["clase"].value_counts()
print("\nFotos por clase:")
print(conteo)
print("\nSana / enferma:")
print(data["binaria"].value_counts())

mayor, menor = conteo.max(), conteo.min()
print(f"\nLa clase más grande tiene {mayor} fotos y la más pequeña {menor}: "
      f"{mayor / menor:.0f} veces más.")

# 4. Verificar que las fotos existan
data["existe"] = data["archivo"].apply(lambda f: (FOTOS / f).exists())
print(f"\nFotos que faltan en la carpeta Photos: {(~data['existe']).sum()}")
print(f"Archivos repetidos en el Excel: {data['archivo'].duplicated().sum()}")

# Tamaño (ancho x alto) de 5 fotos
for f in data["archivo"].head(5):
    with Image.open(FOTOS / f) as im:
        print(f"{f}: {im.size[0]} x {im.size[1]}")

# 5. Visualizar el número de fotos por clase
plt.figure(figsize=(8, 5))
sns.countplot(data=data, y="clase", order=conteo.index, color="#A5531A")
plt.xlabel("Número de fotos")
plt.ylabel("Clase")
plt.title("Fotos por clase en RoCoLe")
plt.savefig(SALIDAS / "conteo_clases.png", dpi=150, bbox_inches="tight")
plt.show()

# 6. Visualizar ejemplos de cada clase (3 fotos por clase)
N = 3
fig, ejes = plt.subplots(len(conteo), N, figsize=(N * 2.2, len(conteo) * 3.2))
for fila, clase in enumerate(conteo.index):
    muestra = data[data["clase"] == clase].sample(min(N, conteo[clase]),
                                                  random_state=42)
    for col in range(N):
        ax = ejes[fila][col]
        ax.axis("off")
        if col < len(muestra):
            with Image.open(FOTOS / muestra.iloc[col]["archivo"]) as im:
                im.thumbnail((400, 400))
                ax.imshow(im)
            if col == 0:
                ax.set_title(clase, loc="left", fontsize=10)
plt.tight_layout()
plt.savefig(SALIDAS / "ejemplos.png", dpi=110)
plt.show()