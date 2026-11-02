# Paso 9 - ¿Qué pasa si la hoja se recorta de forma automática?
# En la app no habrá contorno dibujado a mano. Aquí se prueban dos formas
# automáticas y se mide cuánto baja el acierto.

# 1. Importar librerías necesarias
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

import caracteristicas as car

# Para una prueba rápida ponga un número (por ejemplo 300); None = todas las fotos
N_FOTOS = None

# 2. Carga de datos
RAIZ = Path(__file__).resolve().parent
DATOS = RAIZ / "data" / "RoCoLe A robusta coffee leaf images dataset"
FOTOS = DATOS / "Photos"
CSV_CONTORNOS = DATOS / "Annotations" / "RoCoLE-csv.csv"

data = pd.read_csv(RAIZ / "data.csv")
data["planta"] = data["archivo"].str.extract(r"^(C\d+P\d+)")[0]
if N_FOTOS:
    data = data.sample(N_FOTOS, random_state=42).reset_index(drop=True)

contornos = {}
for _, fila in pd.read_csv(CSV_CONTORNOS).iterrows():
    try:
        etiqueta = json.loads(fila["Label"])
        contornos[fila["External ID"]] = [(p["x"], p["y"])
                                          for p in etiqueta["Leaf"][0]["geometry"]]
    except (KeyError, IndexError, TypeError, ValueError):
        pass


# 3. Dos formas automáticas de aislar la hoja
def mascara_centro(alto, ancho):
    """Rectángulo en el centro de la foto (60% del ancho y del alto)."""
    mascara = np.zeros((alto, ancho), dtype=bool)
    mascara[int(alto * 0.2):int(alto * 0.8), int(ancho * 0.2):int(ancho * 0.8)] = True
    return mascara


def mascara_grabcut(imagen_rgb):
    """GrabCut: separa el objeto del centro del fondo, usando el color."""
    arreglo = np.asarray(imagen_rgb)
    alto, ancho = arreglo.shape[:2]
    chica = cv2.resize(arreglo, (512, int(alto * 512 / ancho)))
    chica = cv2.cvtColor(chica, cv2.COLOR_RGB2BGR)
    h, w = chica.shape[:2]
    resultado = np.zeros((h, w), np.uint8)
    fondo = np.zeros((1, 65))
    frente = np.zeros((1, 65))
    rectangulo = (int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8))
    cv2.grabCut(chica, resultado, rectangulo, fondo, frente, 3, cv2.GC_INIT_WITH_RECT)
    hoja = ((resultado == cv2.GC_FGD) | (resultado == cv2.GC_PR_FGD)).astype(np.uint8)
    hoja = cv2.resize(hoja, (ancho, alto), interpolation=cv2.INTER_NEAREST).astype(bool)
    if hoja.mean() < 0.05 or hoja.mean() > 0.9:   # falló: usar el centro
        return mascara_centro(alto, ancho)
    return hoja


# 4. Extraer características con cada máscara
filas_centro, filas_grabcut, ious, errores = [], [], [], []
for numero, archivo in enumerate(data["archivo"], start=1):
    try:
        with Image.open(FOTOS / archivo) as foto:
            imagen, factor = car.reducir(foto)
        ancho, alto = imagen.size

        # Máscara de referencia (contorno dibujado) solo para medir el parecido
        referencia = Image.new("L", imagen.size, 0)
        puntos = [(x * factor, y * factor) for x, y in contornos[archivo]]
        ImageDraw.Draw(referencia).polygon(puntos, fill=1)
        referencia = np.asarray(referencia, dtype=bool)

        m_centro = mascara_centro(alto, ancho)
        m_grabcut = mascara_grabcut(imagen)

        fila_centro = [archivo] + car.extraer(imagen, m_centro)
        fila_grabcut = [archivo] + car.extraer(imagen, m_grabcut)
        filas_centro.append(fila_centro)
        filas_grabcut.append(fila_grabcut)

        iou = lambda m: (m & referencia).sum() / (m | referencia).sum()
        ious.append((iou(m_centro), iou(m_grabcut)))
    except Exception as error:
        errores.append((archivo, str(error)))
    if numero % 100 == 0:
        print(f"{numero} de {len(data)} fotos procesadas")

columnas = ["archivo"] + car.NOMBRES
tabla_centro = pd.DataFrame(filas_centro, columns=columnas).set_index("archivo")
tabla_grabcut = pd.DataFrame(filas_grabcut, columns=columnas).set_index("archivo")
tabla_centro.to_csv(RAIZ / "data_centro.csv")
tabla_grabcut.to_csv(RAIZ / "data_grabcut.csv")

ious = np.array(ious)
print(f"\nFotos con error: {len(errores)}")
print("Parecido con el contorno real (1.0 = igual, 0 = nada en común):")
print(f"  Centro:   {ious[:, 0].mean():.2f}")
print(f"  GrabCut:  {ious[:, 1].mean():.2f}")

# 5. Medir el acierto: validación cruzada por planta
base = data.set_index("archivo")
modelo = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
print("\nAcierto sana/enferma (validación cruzada por planta):")
for nombre, tabla in [("Centro", tabla_centro), ("GrabCut", tabla_grabcut)]:
    X_contorno = base.loc[tabla.index, car.NOMBRES]   # características con contorno
    X_auto = tabla[car.NOMBRES]                        # características automáticas
    y = base.loc[tabla.index, "target"]
    grupos = base.loc[tabla.index, "planta"]
    ref, mismo, cruzado = [], [], []
    for entrena, prueba in GroupKFold(n_splits=5).split(X_auto, y, groups=grupos):
        # a) todo con contorno (la cifra de antes, para comparar)
        modelo.fit(X_contorno.iloc[entrena], y.iloc[entrena])
        ref.append(modelo.score(X_contorno.iloc[prueba], y.iloc[prueba]))
        # b) entrenar y probar con la máscara automática
        modelo.fit(X_auto.iloc[entrena], y.iloc[entrena])
        mismo.append(modelo.score(X_auto.iloc[prueba], y.iloc[prueba]))
        # c) entrenar con contorno y probar con la automática
        modelo.fit(X_contorno.iloc[entrena], y.iloc[entrena])
        cruzado.append(modelo.score(X_auto.iloc[prueba], y.iloc[prueba]))
    print(f"\n  Máscara {nombre}:")
    print(f"    Todo con contorno (referencia):          {np.mean(ref) * 100:.2f}%")
    print(f"    Entrenar y probar con máscara {nombre}: {np.mean(mismo) * 100:.2f}%")
    print(f"    Entrenar con contorno, probar con {nombre}: {np.mean(cruzado) * 100:.2f}%")