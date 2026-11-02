# Funciones para convertir una foto de hoja en números (características).
# Este archivo se usa igual al entrenar (en el computador) y en la app (en Render).

import numpy as np
import cv2
from PIL import Image
from scipy import ndimage
from sklearn.cluster import KMeans

ANCHO_TRABAJO = 1024  # las fotos se reducen a este ancho para ir más rápido
AREA_MINIMA_MANCHA = 30  # píxeles; manchas más pequeñas se ignoran como ruido

NOMBRES = [
    "media_R", "media_G", "media_B", "std_R", "std_G", "std_B",
    "media_H", "media_S", "media_V", "std_H", "std_S", "std_V",
    "fraccion_amarillo_naranja",
    "porcion_verde", "porcion_medio", "porcion_menos_verde",
    "menos_verde_R", "menos_verde_G", "menos_verde_B",
    "area_hoja_pct",
    # Características de manchas (nuevas)
    "n_manchas", "area_mancha_media", "area_mancha_mayor",
    "fraccion_manchas", "fraccion_marron", "textura",
]


def reducir(imagen):
    """Reduce la foto al ancho de trabajo y la devuelve en RGB."""
    imagen = imagen.convert("RGB")
    factor = ANCHO_TRABAJO / imagen.size[0]
    nuevo = (ANCHO_TRABAJO, int(imagen.size[1] * factor))
    return imagen.resize(nuevo), factor



def mascara_centro(alto, ancho):
    """Rectángulo en el centro de la foto (60% del ancho y del alto)."""
    mascara = np.zeros((alto, ancho), dtype=bool)
    mascara[int(alto * 0.2):int(alto * 0.8), int(ancho * 0.2):int(ancho * 0.8)] = True
    return mascara


def mascara_grabcut(imagen_rgb):
    """Aísla la hoja con GrabCut (sin contorno dibujado). Es la máscara de la app."""
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





def segmentar_manchas(imagen_rgb, mascara, k=5, voto_minimo=0.3, espacio="lab"):
    """
    K-Means sobre el color de la hoja. Devuelve (mapa, porcentaje):
    mapa = True en los píxeles de grupos de color afectado (amarillo, naranja o café);
    porcentaje = qué parte de la hoja ocupa esa zona.

    espacio="lab": agrupa solo por tono (canales a y b), sin mirar el brillo, para que
                   las sombras y el sol no armen los grupos. Es lo que usa la app.
    espacio="rgb": agrupa por RGB (versión anterior, para comparar).
    Un grupo se marca como afectado si al menos `voto_minimo` de sus píxeles
    son amarillos, naranjas o café.
    """
    rgb = np.asarray(imagen_rgb)
    if espacio == "lab":
        datos = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(float)[mascara][:, 1:3]
    else:
        datos = rgb.astype(float)[mascara]

    rng = np.random.RandomState(42)
    muestra = datos[rng.choice(len(datos), size=min(5000, len(datos)), replace=False)]
    kmeans = KMeans(n_clusters=k, n_init=3, random_state=42).fit(muestra)
    etiquetas = kmeans.predict(datos)

    # Qué píxeles tienen color de zona afectada (la misma regla de las características)
    hsv = np.asarray(imagen_rgb.convert("HSV"), dtype=float)[mascara]
    tono, saturacion, brillo = hsv.T
    color_afectado = (tono >= 10) & (tono <= 45) & (saturacion > 64) & (brillo > 50)

    # Cada grupo "vota": si suficientes píxeles son de color afectado, el grupo es afectado
    afectado = np.array([color_afectado[etiquetas == g].mean() >= voto_minimo
                         if (etiquetas == g).any() else False for g in range(k)])

    mapa = np.zeros(mascara.shape, dtype=bool)
    mapa[mascara] = afectado[etiquetas]
    return mapa, mapa.sum() / mascara.sum() * 100






def extraer(imagen_rgb, mascara):
    """
    Calcula las características de los píxeles de la hoja.

    imagen_rgb: imagen PIL en RGB (ya reducida)
    mascara: arreglo de True/False del mismo tamaño; True = píxel de la hoja
    """
    rgb = np.asarray(imagen_rgb, dtype=float)
    hsv = np.asarray(imagen_rgb.convert("HSV"), dtype=float)

    pixeles_rgb = rgb[mascara]
    pixeles_hsv = hsv[mascara]
    if len(pixeles_rgb) < 50:
        raise ValueError("La máscara casi no tiene píxeles de hoja")

    caracteristicas = {}

    # Color promedio y variación (RGB y HSV)
    for i, canal in enumerate("RGB"):
        caracteristicas[f"media_{canal}"] = pixeles_rgb[:, i].mean()
        caracteristicas[f"std_{canal}"] = pixeles_rgb[:, i].std()
    for i, canal in enumerate("HSV"):
        caracteristicas[f"media_{canal}"] = pixeles_hsv[:, i].mean()
        caracteristicas[f"std_{canal}"] = pixeles_hsv[:, i].std()

    # Fracción de píxeles amarillos, naranjas o café (posibles manchas)
    tono, saturacion, brillo = pixeles_hsv.T
    amarillo_naranja = (tono >= 10) & (tono <= 45) & (saturacion > 64) & (brillo > 50)
    caracteristicas["fraccion_amarillo_naranja"] = amarillo_naranja.mean()

    # K-Means con 3 grupos de color dentro de la hoja
    rng = np.random.RandomState(42)
    muestra = pixeles_rgb[rng.choice(len(pixeles_rgb),
                                     size=min(4000, len(pixeles_rgb)),
                                     replace=False)]
    kmeans = KMeans(n_clusters=3, n_init=3, random_state=42).fit(muestra)
    centros = kmeans.cluster_centers_
    grupos = kmeans.predict(pixeles_rgb)
    # Ordenar los grupos del más verde al menos verde
    verdor = centros[:, 1] - (centros[:, 0] + centros[:, 2]) / 2
    orden = np.argsort(-verdor)
    porciones = [(grupos == g).mean() for g in orden]
    caracteristicas["porcion_verde"] = porciones[0]
    caracteristicas["porcion_medio"] = porciones[1]
    caracteristicas["porcion_menos_verde"] = porciones[2]
    menos_verde = centros[orden[2]]
    caracteristicas["menos_verde_R"] = menos_verde[0]
    caracteristicas["menos_verde_G"] = menos_verde[1]
    caracteristicas["menos_verde_B"] = menos_verde[2]

    # Qué parte de la foto ocupa la hoja
    caracteristicas["area_hoja_pct"] = mascara.mean() * 100

    # ---- Manchas: cuántas son, qué tan grandes y cuánto cubren de la hoja
    area_hoja = mascara.sum()
    tono2, sat2, brillo2 = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    manchas = mascara & (tono2 >= 10) & (tono2 <= 45) & (sat2 > 64) & (brillo2 > 50)
    manchas = ndimage.binary_opening(manchas, structure=np.ones((3, 3)))
    etiquetas, _ = ndimage.label(manchas)
    areas = np.bincount(etiquetas.ravel())[1:]
    areas = areas[areas >= AREA_MINIMA_MANCHA]

    caracteristicas["n_manchas"] = len(areas)
    caracteristicas["area_mancha_media"] = (areas.mean() / area_hoja * 100) if len(areas) else 0.0
    caracteristicas["area_mancha_mayor"] = (areas.max() / area_hoja * 100) if len(areas) else 0.0
    caracteristicas["fraccion_manchas"] = areas.sum() / area_hoja

    # Zonas café oscuras (tejido muerto)
    marron = mascara & (tono2 <= 30) & (brillo2 < 140) & (sat2 > 80)
    caracteristicas["fraccion_marron"] = marron.sum() / area_hoja

    # Textura: cuánto cambia el brillo de un píxel al vecino (sin contar el borde)
    gris = rgb.mean(axis=2)
    gy, gx = np.gradient(gris)
    cambio = np.hypot(gx, gy)
    interior = ndimage.binary_erosion(mascara, iterations=3)
    if interior.sum() < 50:
        interior = mascara
    caracteristicas["textura"] = cambio[interior].mean()

    return [float(caracteristicas[n]) for n in NOMBRES]