# Ojo al Cafetal - app web (Flask)
# Recibe la foto de una hoja y dice si está sana o enferma.
# El modelo se entrenó en el computador; aquí solo se carga y se usa.

import base64
import io
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from PIL import Image, ImageOps

import caracteristicas as car

RAIZ = Path(__file__).resolve().parent
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # fotos de hasta 10 MB

# Cargar el modelo una sola vez, al iniciar la app
paquete = joblib.load(RAIZ / "models" / "modelo_app.joblib")
modelo = paquete["modelo"]
columnas = paquete["columnas"]

# Modelo de Q-Learning (entrenado en reinforcement.py): recomienda qué hacer
q = joblib.load(RAIZ / "models" / "qlearning.joblib")
Q_PESOS = q["q_model"].coef_
ACCIONES = q["acciones"]
N_NIVELES = len(q["niveles"])
N_ACCIONES = len(ACCIONES)

TEXTO_ACCION = {
    "Solo monitorear": "Siga revisando las hojas cada mes; por ahora no hace falta tratar.",
    "Mejorar el manejo del cultivo": "Mejore el manejo: fertilización según análisis de suelo, "
                                     "control de arvenses y de sombra, y densidad adecuada.",
    "Aplicar fungicida": "Consulte a un técnico para aplicar un fungicida sistémico a tiempo.",
}


def nivel_estimado(enferma, afectada):
    """Nivel de roya aproximado (0 a 4) según la zona marcada por K-Means."""
    if not enferma:
        return 0
    if afectada < 3:
        return 1
    if afectada < 8:
        return 2
    if afectada < 15:
        return 3
    return 4


def recomendar(nivel, variedad):
    """Mejor acción según Q-Learning. variedad: 0 = resistente, 1 = susceptible."""
    estado = variedad * N_NIVELES + nivel
    valores = Q_PESOS[estado * N_ACCIONES:(estado + 1) * N_ACCIONES]
    accion = ACCIONES[int(np.argmax(valores))]
    texto = TEXTO_ACCION[accion]
    if accion == "Aplicar fungicida" and variedad == 0:
        texto = ("Es poco común en variedades resistentes: consulte primero a un técnico "
                 "antes de aplicar nada.")
    return {"variedad": ["resistente", "susceptible"][variedad], "accion": accion, "texto": texto}


def recorte_dudoso(mascara):
    """True si el recorte parece no ser una sola hoja:
    cubre casi toda la foto o tiene forma muy irregular."""
    m = mascara.astype(np.uint8)
    fraccion = m.mean()
    contornos, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return True
    mayor = max(contornos, key=cv2.contourArea)
    casco = cv2.contourArea(cv2.convexHull(mayor))
    solidez = cv2.contourArea(mayor) / casco if casco > 0 else 0
    return fraccion > 0.55 or solidez < 0.80


def foto_con_contorno(imagen, mascara, mapa=None):
    """Devuelve la foto con la zona analizada (línea amarilla) y, si se da,
    la zona afectada (magenta), como texto base64."""
    arreglo = np.asarray(imagen).copy()
    if mapa is not None:
        magenta = np.array([255, 0, 255], dtype=float)
        arreglo[mapa] = (0.55 * arreglo[mapa] + 0.45 * magenta).astype(np.uint8)
    contornos, _ = cv2.findContours(mascara.astype(np.uint8), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(arreglo, contornos, -1, (255, 255, 0), 4)
    marcada =