# Ojo al Cafetal - web app (Flask)
# Receives a photo of a coffee leaf and says if it looks healthy or sick.
# The model was trained on a computer; here it is only loaded and used.

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
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # photos up to 10 MB

# Load the model once, when the app starts
paquete = joblib.load(RAIZ / "models" / "modelo_app.joblib")
modelo = paquete["modelo"]
columnas = paquete["columnas"]

# Q-Learning model (trained in reinforcement.py): recommends what to do
q = joblib.load(RAIZ / "models" / "qlearning.joblib")
Q_PESOS = q["q_model"].coef_
ACCIONES = q["acciones"]
N_NIVELES = len(q["niveles"])
N_ACCIONES = len(ACCIONES)

# The saved model uses Spanish action names; these are the English texts shown to the user
NOMBRE_ACCION = {
    "Solo monitorear": "Keep monitoring",
    "Mejorar el manejo del cultivo": "Improve crop management",
    "Aplicar fungicida": "Apply fungicide",
}
TEXTO_ACCION = {
    "Solo monitorear": "Keep checking the leaves every month. No treatment is needed for now.",
    "Mejorar el manejo del cultivo": "Improve management: fertilize based on a soil test, control weeds "
                                     "and shade, and keep a good plant density.",
    "Aplicar fungicida": "Ask a technician about applying a systemic fungicide in time.",
}


def nivel_estimado(enferma, afectada):
    """Approximate rust level (0 to 4) from the zone marked by K-Means."""
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
    """Best action from Q-Learning. variedad: 0 = resistant, 1 = susceptible."""
    estado = variedad * N_NIVELES + nivel
    valores = Q_PESOS[estado * N_ACCIONES:(estado + 1) * N_ACCIONES]
    accion = ACCIONES[int(np.argmax(valores))]
    texto = TEXTO_ACCION[accion]
    if accion == "Aplicar fungicida" and variedad == 0:
        texto = ("This is uncommon in resistant varieties: talk to a technician first, "
                 "before applying anything.")
    return {"variedad": ["resistant", "susceptible"][variedad],
            "accion": NOMBRE_ACCION[accion], "texto": texto}


def recorte_dudoso(mascara):
    """True if the cut does not look like a single leaf:
    it covers almost all the photo or has a very irregular shape."""
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
    """Returns the photo with the analyzed zone (yellow line) and, if given,
    the affected zone (magenta), as base64 text."""
    arreglo = np.asarray(imagen).copy()
    if mapa is not None:
        magenta = np.array([255, 0, 255], dtype=float)
        arreglo[mapa] = (0.55 * arreglo[mapa] + 0.45 * magenta).astype(np.uint8)
    contornos, _ = cv2.findContours(mascara.astype(np.uint8), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(arreglo, contornos, -1, (255, 255, 0), 4)
    marcada = Image.fromarray(arreglo)
    marcada.thumbnail((700, 700))
    buffer = io.BytesIO()
    marcada.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@app.route("/")
def inicio():
    return render_template("index.html", resultado=None, error=None, variedad="no_se")


@app.route("/predecir", methods=["POST"])
def predecir():
    archivo = request.files.get("foto")
    variedad = request.form.get("variedad", "no_se")
    if variedad not in ("no_se", "resistente", "susceptible"):
        variedad = "no_se"
    if archivo is None or archivo.filename == "":
        return render_template("index.html", resultado=None, variedad=variedad,
                               error="Choose a photo of a leaf.")
    try:
        foto = Image.open(archivo.stream)
        foto = ImageOps.exif_transpose(foto)          # respect the phone's rotation
        imagen, _ = car.reducir(foto)
        mascara = car.mascara_grabcut(imagen)
        valores = car.extraer(imagen, mascara)
    except Exception:
        return render_template("index.html", resultado=None, variedad=variedad,
                               error="I could not read that photo. Try another image (JPG or PNG).")

    probabilidad = float(modelo.predict_proba(pd.DataFrame([valores], columns=columnas))[0][1])   # probability of "sick"
    enferma = probabilidad >= 0.5
    seguridad = probabilidad if enferma else 1 - probabilidad
    if seguridad >= 0.80:
        nivel = "high"
    elif seguridad >= 0.65:
        nivel = "medium"
    else:
        nivel = "low"
    # If the leaf looks sick, K-Means marks the zone with signs of damage
    mapa, afectada = (None, None)
    if enferma:
        mapa, afectada = car.segmentar_manchas(imagen, mascara)
        afectada = round(afectada, 1)
    # Q-Learning: what to do, based on the estimated level and the variety
    nivel_roya = nivel_estimado(enferma, afectada)
    if variedad == "resistente":
        recomendaciones = [recomendar(nivel_roya, 0)]
    elif variedad == "susceptible":
        recomendaciones = [recomendar(nivel_roya, 1)]
    else:
        recomendaciones = [recomendar(nivel_roya, 0), recomendar(nivel_roya, 1)]
    resultado = {
        "enferma": enferma,
        "porcentaje": round(seguridad * 100),
        "nivel": nivel,
        "afectada": afectada,
        "nivel_roya": nivel_roya,
        "recomendaciones": recomendaciones,
        "imagen": foto_con_contorno(imagen, mascara, mapa),
        "dudosa": recorte_dudoso(mascara),
    }
    return render_template("index.html", resultado=resultado, error=None, variedad=variedad)


if __name__ == "__main__":
    app.run(debug=True)