# Paso 11 - K-Means: marcar la zona afectada de la hoja

# 1. Importar librerías necesarias
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import caracteristicas as car

FOTOS_POR_CLASE = 30   # cuántas fotos de cada clase se revisan

# 2. Carga de datos
RAIZ = Path(__file__).resolve().parent
DATOS = RAIZ / "data" / "RoCoLe A robusta coffee leaf images dataset"
FOTOS = DATOS / "Photos"
CSV_CONTORNOS = DATOS / "Annotations" / "RoCoLE-csv.csv"
SALIDAS = RAIZ / "salidas"
SALIDAS.mkdir(exist_ok=True)

data = pd.read_csv(RAIZ / "data.csv")[["archivo", "clase"]]

contornos = {}
for _, fila in pd.read_csv(CSV_CONTORNOS).iterrows():
    try:
        etiqueta = json.loads(fila["Label"])
        contornos[fila["External ID"]] = [(p["x"], p["y"])
                                          for p in etiqueta["Leaf"][0]["geometry"]]
    except (KeyError, IndexError, TypeError, ValueError):
        pass

# 3. Aplicar K-Means a una muestra de fotos de cada clase
# (aquí se usa el contorno dibujado para medir limpio; la app usa GrabCut)
resultados = []
ejemplos = {}
for clase, grupo in data.groupby("clase"):
    muestra = grupo.sample(min(FOTOS_POR_CLASE, len(grupo)), random_state=42)
    for archivo in muestra["archivo"]:
        with Image.open(FOTOS / archivo) as foto:
            imagen, factor = car.reducir(foto)
        mascara_img = Image.new("L", imagen.size, 0)
        puntos = [(x * factor, y * factor) for x, y in contornos[archivo]]
        ImageDraw.Draw(mascara_img).polygon(puntos, fill=1)
        mascara = np.asarray(mascara_img, dtype=bool)

        mapa_rgb, pct_rgb = car.segmentar_manchas(imagen, mascara, k=4, voto_minimo=0.35, espacio="rgb")
        mapa, porcentaje = car.segmentar_manchas(imagen, mascara)   # versión nueva (Lab)
        resultados.append((clase, archivo, pct_rgb, porcentaje))
        if clase not in ejemplos:
            ejemplos[clase] = (imagen, mascara, mapa, porcentaje)
    print(f"{clase}: listo")

# 4. Porcentaje afectado por clase
tabla = pd.DataFrame(resultados, columns=["clase", "archivo", "pct_rgb", "porcentaje"])
orden = [c for c in ["healthy", "rust_level_1", "rust_level_2", "rust_level_3",
                     "rust_level_4", "red_spider_mite"] if c in ejemplos]
resumen = pd.DataFrame({
    "RGB_prom": tabla.groupby("clase")["pct_rgb"].mean(),
    "RGB_mediana": tabla.groupby("clase")["pct_rgb"].median(),
    "Lab_prom": tabla.groupby("clase")["porcentaje"].mean(),
    "Lab_mediana": tabla.groupby("clase")["porcentaje"].median(),
    "fotos": tabla.groupby("clase")["porcentaje"].count(),
}).loc[orden]
print("\nZona afectada de la hoja (%), por clase: RGB = versión anterior, Lab = versión nueva")
print(resumen.round(1).to_string())

# 5. Imagen de ejemplo: una foto por clase, con la zona afectada en magenta
fig, ejes = plt.subplots(len(orden), 2, figsize=(7, len(orden) * 2.4))
for fila, clase in enumerate(orden):
    imagen, mascara, mapa, porcentaje = ejemplos[clase]
    arreglo = np.asarray(imagen).copy()
    marcada = arreglo.copy()
    marcada[mapa] = (255, 0, 255)
    marcada[~mascara] = (marcada[~mascara] * 0.3).astype("uint8")   # oscurecer el fondo
    ejes[fila][0].imshow(arreglo)
    ejes[fila][1].imshow(marcada)
    ejes[fila][0].set_title(clase, loc="left", fontsize=9)
    ejes[fila][1].set_title(f"afectada: {porcentaje:.1f}%", loc="left", fontsize=9)
    ejes[fila][0].axis("off")
    ejes[fila][1].axis("off")
fig.tight_layout()
fig.savefig(SALIDAS / "kmeans_ejemplos.png", dpi=110)
plt.show()
print("\nImagen guardada en salidas/kmeans_ejemplos.png")