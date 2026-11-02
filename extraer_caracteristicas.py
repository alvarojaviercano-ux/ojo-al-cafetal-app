# Paso 2 - Convertir las fotos en una tabla de números (data.csv)

# 1. Importar librerías necesarias
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import caracteristicas as car

# 2. Carga de datos
RAIZ = Path(__file__).resolve().parent
DATOS = RAIZ / "data" / "RoCoLe A robusta coffee leaf images dataset"
FOTOS = DATOS / "Photos"
EXCEL = DATOS / "Annotations" / "RoCoLe-classes.xlsx"
CSV_CONTORNOS = DATOS / "Annotations" / "RoCoLE-csv.csv"

etiquetas = pd.read_excel(EXCEL)
etiquetas = etiquetas.rename(columns={"File": "archivo",
                                      "Binary.Label": "binaria",
                                      "Multiclass.Label": "clase"})

# El CSV de anotaciones trae el contorno de la hoja de cada foto
anotaciones = pd.read_csv(CSV_CONTORNOS)
print(anotaciones[["External ID"]].head())

contornos = {}
for _, fila in anotaciones.iterrows():
    try:
        etiqueta = json.loads(fila["Label"])
        puntos = [(p["x"], p["y"]) for p in etiqueta["Leaf"][0]["geometry"]]
        contornos[fila["External ID"]] = puntos
    except (KeyError, IndexError, TypeError, ValueError):
        pass  # foto sin contorno
print(f"Fotos con contorno de hoja: {len(contornos)} de {len(anotaciones)}")

# 3. Extraer las características de cada foto
filas = []
errores = []
for numero, fila in enumerate(etiquetas.itertuples(), start=1):
    try:
        with Image.open(FOTOS / fila.archivo) as foto:
            # Se usa la foto tal como la lee Python (2048 x 1152): en ese
            # tamaño están los contornos de las anotaciones
            imagen, factor = car.reducir(foto)

        # Crear la máscara de la hoja a partir del contorno
        mascara_img = Image.new("L", imagen.size, 0)
        if fila.archivo in contornos:
            puntos = [(x * factor, y * factor) for x, y in contornos[fila.archivo]]
            ImageDraw.Draw(mascara_img).polygon(puntos, fill=1)
        else:
            # Sin contorno: se usa toda la foto
            mascara_img = Image.new("L", imagen.size, 1)
        mascara = np.asarray(mascara_img, dtype=bool)

        valores = car.extraer(imagen, mascara)
        filas.append([fila.archivo] + valores + [fila.binaria, fila.clase])
    except Exception as error:
        errores.append((fila.archivo, str(error)))

    if numero % 100 == 0:
        print(f"{numero} de {len(etiquetas)} fotos procesadas")

# 4. Guardar la tabla
data = pd.DataFrame(filas, columns=["archivo"] + car.NOMBRES + ["binaria", "clase"])
# Columna 'target' como en el ejemplo del profesor (0 = sana, 1 = enferma)
data["target"] = (data["binaria"] == "unhealthy").astype(int)
data.to_csv(RAIZ / "data.csv", index=False)

print(f"\nTabla guardada: {data.shape[0]} filas y {data.shape[1]} columnas")
print(f"Fotos con error: {len(errores)}")
for archivo, mensaje in errores[:10]:
    print(f"  {archivo}: {mensaje}")

# Explorar la tabla
print(data.head())
print(data.describe())
