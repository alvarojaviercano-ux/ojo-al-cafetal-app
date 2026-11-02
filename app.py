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


def foto_con_contorno(imagen, mascara):
    """Devuelve la foto con la zona analizada marcada, como texto base64."""
    arreglo = np.asarray(imagen).copy()
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
    return render_template("index.html", resultado=None, error=None)


@app.route("/predecir", methods=["POST"])
def predecir():
    archivo = request.files.get("foto")
    if archivo is None or archivo.filename == "":
        return render_template("index.html", resultado=None,
                               error="Elija una foto de una hoja.")
    try:
        foto = Image.open(archivo.stream)
        foto = ImageOps.exif_transpose(foto)          # respetar la rotación del celular
        imagen, _ = car.reducir(foto)
        mascara = car.mascara_grabcut(imagen)
        valores = car.extraer(imagen, mascara)
    except Exception:
        return render_template("index.html", resultado=None,
                               error="No pude leer esa foto. Pruebe con otra imagen (JPG o PNG).")

    probabilidad = float(modelo.predict_proba(pd.DataFrame([valores], columns=columnas))[0][1])   # probabilidad de "enferma"
    enferma = probabilidad >= 0.5
    seguridad = probabilidad if enferma else 1 - probabilidad
    if seguridad >= 0.80:
        nivel = "alta"
    elif seguridad >= 0.65:
        nivel = "media"
    else:
        nivel = "baja"
    resultado = {
        "enferma": enferma,
        "porcentaje": round(seguridad * 100),
        "nivel": nivel,
        "imagen": foto_con_contorno(imagen, mascara),
    }
    return render_template("index.html", resultado=resultado, error=None)


if __name__ == "__main__":
    app.run(debug=True)