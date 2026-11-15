# Eye on the Coffee Farm - information system (Flask)
# Manages coffee farms, plots, leaf inspections and follow-up actions.
# The trained model (Random Forest) is only loaded and used here; it is trained on a computer.
#
# Roles: admin, technician, grower. Everyone who signs up is a grower;
# only an admin can give another role.

import base64
import csv
import hashlib
import io
import json
import logging
import os
import re
import secrets
import time
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from flask import (Flask, Response, abort, flash, g, has_app_context, redirect,
                   render_template, request, session, url_for)
from PIL import Image, ImageOps
from werkzeug.security import check_password_hash, generate_password_hash

import caracteristicas as car
import database as database
from database import now

RAIZ = Path(__file__).resolve().parent

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024          # photos up to 10 MB
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = bool(os.environ.get("RENDER"))   # https on Render
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=8)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cafetal")

MODEL_FILE = RAIZ / "models" / "modelo_app.joblib"
ROLES = ("admin", "technician", "grower")
ROLE_TEXT = {"admin": "Administrator", "technician": "Technician", "grower": "Coffee grower"}
VARIETIES = {"resistant": "Resistant to rust", "susceptible": "Susceptible to rust",
             "unknown": "I do not know"}
LIMITE_FOTOS_POR_HORA = 40
USUARIO_VALIDO = re.compile(r"^[A-Za-z0-9_]{3,30}$")


# ------------------------------------------------------------------ startup
def iniciar():
    """Create tables, the secret key and the accounts given by environment variables."""
    database.crear_tablas()
    conn = database.Conn()
    try:
        fila = conn.one("SELECT value FROM settings WHERE name = 'secret_key'")
        if fila is None:
            conn.run("INSERT INTO settings (name, value) VALUES ('secret_key', ?)",
                     (secrets.token_hex(32),))
            fila = conn.one("SELECT value FROM settings WHERE name = 'secret_key'")
        app.secret_key = os.environ.get("SECRET_KEY") or fila["value"]
        # Accounts from environment variables (no passwords in the code)
        for prefijo, rol in (("ADMIN", "admin"), ("TECHNICIAN", "technician"), ("GROWER", "grower")):
            nombre = os.environ.get(f"{prefijo}_USER", "").strip()
            clave = os.environ.get(f"{prefijo}_PASSWORD", "")
            if nombre and clave and USUARIO_VALIDO.match(nombre):
                if conn.one("SELECT id FROM users WHERE username_lc = ?", (nombre.lower(),)) is None:
                    conn.insert(
                        "INSERT INTO users (username, username_lc, full_name, password_hash, role, "
                        "active, created_at) VALUES (?, ?, ?, ?, ?, 1, ?)",
                        (nombre, nombre.lower(), ROLE_TEXT[rol] + " (test account)",
                         generate_password_hash(clave), rol, now()))
        # A small demo farm for the grower test account
        nombre = os.environ.get("GROWER_USER", "").strip().lower()
        if nombre:
            dueno = conn.one("SELECT id FROM users WHERE username_lc = ?", (nombre,))
            if dueno and conn.one("SELECT id FROM farms WHERE owner_id = ?", (dueno["id"],)) is None:
                finca = conn.insert(
                    "INSERT INTO farms (name, municipality, department, owner_id, created_at) "
                    "VALUES (?, ?, ?, ?, ?)", ("Demo farm", "Fusagasugá", "Cundinamarca", dueno["id"], now()))
                conn.insert("INSERT INTO plots (farm_id, name, variety, area_ha, planted_year, notes, "
                            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (finca, "Plot 1", "susceptible", 1.5, 2021, "Demo plot", now()))
        conn.commit()
    finally:
        conn.close()


iniciar()
app.teardown_request(database.cerrar_request)


# --------------------------------------------------------------------- model
paquete = joblib.load(MODEL_FILE)
modelo = paquete["modelo"]
columnas = paquete["columnas"]
_huella = hashlib.sha1(MODEL_FILE.read_bytes()).hexdigest()[:8]
MODEL_VERSION = f"{paquete.get('version', 'RandomForest-300')} · {_huella}"

q = joblib.load(RAIZ / "models" / "qlearning.joblib")
Q_PESOS = q["q_model"].coef_
ACCIONES = q["acciones"]
N_NIVELES = len(q["niveles"])
N_ACCIONES = len(ACCIONES)

try:
    REFERENCIA = json.loads((RAIZ / "models" / "referencia_monitoreo.json").read_text())
except Exception:
    REFERENCIA = None

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
    """True if the cut does not look like a single leaf."""
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
    """Photo with the analyzed zone (yellow line) and the affected zone (magenta), as base64."""
    arreglo = np.asarray(imagen).copy()
    if mapa is not None:
        magenta = np.array([255, 0, 255], dtype=float)
        arreglo[mapa] = (0.55 * arreglo[mapa] + 0.45 * magenta).astype(np.uint8)
    contornos, _ = cv2.findContours(mascara.astype(np.uint8), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(arreglo, contornos, -1, (255, 255, 0), 4)
    marcada = Image.fromarray(arreglo)
    marcada.thumbnail((640, 640))
    buffer = io.BytesIO()
    marcada.save(buffer, format="JPEG", quality=80)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def analizar(archivo, variedad_lote):
    """Runs the whole model pipeline on one uploaded photo. Raises ValueError if unreadable."""
    inicio = time.perf_counter()
    try:
        foto = Image.open(archivo.stream)
        foto = ImageOps.exif_transpose(foto)
        imagen, _ = car.reducir(foto)
        mascara = car.mascara_grabcut(imagen)
        valores = car.extraer(imagen, mascara)
    except Exception as e:
        raise ValueError("I could not read that photo. Try another image (JPG or PNG).") from e
    probabilidad = float(modelo.predict_proba(pd.DataFrame([valores], columns=columnas))[0][1])
    enferma = probabilidad >= 0.5
    seguridad = probabilidad if enferma else 1 - probabilidad
    nivel = "high" if seguridad >= 0.80 else "medium" if seguridad >= 0.65 else "low"
    mapa, afectada = None, None
    if enferma:
        mapa, afectada = car.segmentar_manchas(imagen, mascara)
        afectada = round(float(afectada), 1)
    nivel_roya = nivel_estimado(enferma, afectada)
    if variedad_lote == "resistant":
        recs = [recomendar(nivel_roya, 0)]
    elif variedad_lote == "susceptible":
        recs = [recomendar(nivel_roya, 1)]
    else:
        recs = [recomendar(nivel_roya, 0), recomendar(nivel_roya, 1)]
    return {
        "enferma": bool(enferma), "probabilidad": probabilidad, "porcentaje": round(seguridad * 100),
        "nivel": nivel, "afectada": afectada, "nivel_roya": nivel_roya,
        "recomendaciones": recs if enferma else [],
        "imagen": foto_con_contorno(imagen, mascara, mapa),
        "dudosa": bool(recorte_dudoso(mascara)),
        "valores": [float(v) for v in valores],
        "ms": int((time.perf_counter() - inicio) * 1000),
    }


# ----------------------------------------------------------- users and logs
def db():
    return database.get()


def usuario():
    """Logged-in user (a dict) or None. Looked up in the database at every request,
    so a disabled account or a changed role takes effect immediately."""
    if "user" not in g:
        g.user = None
        uid = session.get("uid")
        if uid is not None:
            g.user = db().one("SELECT id, username, full_name, role FROM users "
                              "WHERE id = ? AND active = 1", (uid,))
    return g.user


def registrar(nivel, evento, detalle="", ms=None):
    """Writes a line in the event log (errors, inferences, logins, admin actions)."""
    try:
        u = usuario() if has_app_context() else None
        db().run("INSERT INTO event_log (ts, level, event, user_id, detail, ms) VALUES (?, ?, ?, ?, ?, ?)",
                 (now(), nivel, evento, u["id"] if u else None, detalle[:500], ms))
    except Exception:
        log.exception("could not write the event log")
    getattr(log, "error" if nivel == "error" else "warning" if nivel == "warning" else "info")(
        "%s %s", evento, detalle)


def requiere_login(f):
    @wraps(f)
    def interior(*a, **k):
        if usuario() is None:
            flash("Log in to continue.")
            return redirect(url_for("login", next=request.path))
        return f(*a, **k)
    return interior


def requiere_rol(*roles):
    def decorador(f):
        @wraps(f)
        def interior(*a, **k):
            u = usuario()
            if u is None:
                flash("Log in to continue.")
                return redirect(url_for("login", next=request.path))
            if u["role"] not in roles:
                registrar("warning", "forbidden", f"{request.method} {request.path}")
                abort(403)
            return f(*a, **k)
        return interior
    return decorador


def es_personal(u=None):
    u = u or usuario()
    return bool(u) and u["role"] in ("admin", "technician")


def token_csrf():
    if "csrf" not in session:
        session["csrf"] = secrets.token_hex(16)
    return session["csrf"]


FASES = [
    ("understanding", "Business and Data Understanding"),
    ("data-engineering", "Data Engineering"),
    ("model-engineering", "Model Engineering"),
    ("evaluation", "Model Evaluation"),
    ("deployment", "Deployment"),
    ("monitoring", "Monitoring and Maintenance"),
]


@app.context_processor
def datos_comunes():
    u = usuario()
    return {"usuario": u["username"] if u else None, "rol": u["role"] if u else None,
            "csrf_token": token_csrf, "FASES": FASES, "fase_actual": None,
            "ROLE_TEXT": ROLE_TEXT, "VARIETIES": VARIETIES, "MODEL_VERSION": MODEL_VERSION}


@app.template_filter("dt")
def filtro_fecha(valor):
    """UTC text -> Colombia time (UTC-5, no daylight saving)."""
    if not valor:
        return "-"
    try:
        t = datetime.strptime(valor, "%Y-%m-%d %H:%M:%S") - timedelta(hours=5)
        return t.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return valor


ESTADOS = {"pending_review": "Waiting for technician review", "action_required": "Action required",
           "closed": "Closed"}
ESTADOS_ACCION = {"pending": "Pending", "in_progress": "In progress", "done": "Done",
                  "dismissed": "Dismissed"}


@app.template_filter("estado")
def filtro_estado(v):
    return ESTADOS.get(v, v)


@app.template_filter("estado_accion")
def filtro_estado_accion(v):
    return ESTADOS_ACCION.get(v, v)


@app.before_request
def revisar_csrf():
    if request.method == "POST":
        enviado = request.form.get("csrf", "")
        if not enviado or not secrets.compare_digest(enviado, session.get("csrf", "")):
            abort(400, "Invalid or expired form. Go back, reload the page and try again.")
    g.t0 = time.perf_counter()


@app.after_request
def cerrar(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "same-origin"
    if request.path.startswith(("/farms", "/plots", "/inspections", "/dashboard", "/admin",
                                "/monitoring", "/reports", "/feedback")):
        resp.headers["Cache-Control"] = "no-store"
    conn = g.get("conn")
    if conn is not None:
        try:
            ms = int((time.perf_counter() - g.get("t0", time.perf_counter())) * 1000)
            if resp.status_code >= 500:
                conn.rollback()
            else:
                if ms > 5000:
                    registrar("warning", "slow_request", f"{request.method} {request.path}", ms)
                conn.commit()
        except Exception:
            conn.rollback()
            log.exception("commit failed")
    return resp


def pagina_error(codigo, titulo, mensaje):
    return render_template("error.html", codigo=codigo, titulo=titulo, mensaje=mensaje), codigo


@app.errorhandler(400)
def e400(e):
    return pagina_error(400, "Something is wrong with that request", str(e.description))


@app.errorhandler(403)
def e403(e):
    return pagina_error(403, "Not allowed", "Your role does not have access to this page or this data.")


@app.errorhandler(404)
def e404(e):
    return pagina_error(404, "Page not found", "That page or record does not exist.")


@app.errorhandler(413)
def e413(e):
    return pagina_error(413, "Photo too big", "Use a photo smaller than 10 MB.")


@app.errorhandler(500)
def e500(e):
    try:
        db().rollback()
        registrar("error", "server_error", f"{request.method} {request.path}: {e!r}")
        db().commit()
    except Exception:
        log.exception("could not log the error")
    return pagina_error(500, "Server error", "Something failed on our side. It was logged. Try again in a moment.")


# ------------------------------------------------------------ public pages
@app.route("/")
def resumen():
    return render_template("resumen.html")


@app.route("/phases/<slug>")
def fase(slug):
    if slug not in dict(FASES):
        abort(404)
    return render_template(f"phases/{slug}.html", fase_actual=slug)


@app.route("/manuals/<tipo>")
def manual(tipo):
    if tipo not in ("user", "technical", "training"):
        abort(404)
    return render_template(f"manual_{tipo}.html")


@app.route("/health")
def salud():
    try:
        db().one("SELECT 1 AS ok")
        return {"status": "ok", "model": MODEL_VERSION}
    except Exception:
        return {"status": "database error"}, 500


# ---------------------------------------------------------- accounts
def siguiente_seguro(destino):
    if destino and destino.startswith("/") and not destino.startswith("//"):
        return destino
    return url_for("panel")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        nombre = request.form.get("usuario", "").strip()
        completo = request.form.get("nombre", "").strip()[:80]
        clave = request.form.get("clave", "")
        repetir = request.form.get("repetir", "")
        if not USUARIO_VALIDO.match(nombre):
            error = "The username must have 3 to 30 characters: letters, numbers or _."
        elif len(clave) < 8 or not re.search(r"[A-Za-z]", clave) or not re.search(r"\d", clave):
            error = "The password must have at least 8 characters, with letters and numbers."
        elif clave != repetir:
            error = "The two passwords are not the same."
        elif db().one("SELECT id FROM users WHERE username_lc = ?", (nombre.lower(),)):
            error = "That username is already taken."
        else:
            # The role is ALWAYS "grower" here. Any "role" field in the form is ignored.
            nuevo = db().insert(
                "INSERT INTO users (username, username_lc, full_name, password_hash, role, active, "
                "created_at) VALUES (?, ?, ?, ?, 'grower', 1, ?)",
                (nombre, nombre.lower(), completo, generate_password_hash(clave), now()))
            session.clear()
            session["uid"] = nuevo
            session.permanent = True
            g.pop("user", None)
            registrar("info", "register", nombre)
            flash("Welcome! Your account was created as a coffee grower. Start by adding a farm.")
            return redirect(url_for("panel"))
        return render_template("register.html", error=error, nombre=nombre, completo=completo), 400
    return render_template("register.html", error=None, nombre="", completo="")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        nombre = request.form.get("usuario", "").strip()
        clave = request.form.get("clave", "")
        fila = db().one("SELECT id, password_hash, active FROM users WHERE username_lc = ?",
                        (nombre.lower(),))
        if fila and fila["active"] and check_password_hash(fila["password_hash"], clave):
            destino = request.args.get("next")
            session.clear()
            session["uid"] = fila["id"]
            session.permanent = True
            g.pop("user", None)
            db().run("UPDATE users SET last_login = ? WHERE id = ?", (now(), fila["id"]))
            registrar("info", "login", nombre)
            return redirect(siguiente_seguro(destino))
        registrar("warning", "login_failed", nombre[:30])
        return render_template("login.html", error="Wrong username or password.", nombre=nombre), 401
    return render_template("login.html", error=None, nombre="")


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("resumen"))


# ----------------------------------------------------------- access rules
def finca_o_error(fid):
    f = db().one("SELECT f.*, u.username AS owner_name FROM farms f JOIN users u ON u.id = f.owner_id "
                 "WHERE f.id = ?", (fid,))
    if f is None:
        abort(404)
    u = usuario()
    if not (es_personal(u) or f["owner_id"] == u["id"]):
        registrar("warning", "forbidden", f"farm {fid}")
        abort(403)
    return f


def lote_o_error(pid):
    p = db().one("SELECT p.*, f.name AS farm_name, f.owner_id, f.id AS farm_id_ FROM plots p "
                 "JOIN farms f ON f.id = p.farm_id WHERE p.id = ?", (pid,))
    if p is None:
        abort(404)
    u = usuario()
    if not (es_personal(u) or p["owner_id"] == u["id"]):
        registrar("warning", "forbidden", f"plot {pid}")
        abort(403)
    return p


def inspeccion_o_error(iid, con_imagen=True):
    campos = "i.*" if con_imagen else ("i.id, i.plot_id, i.created_by, i.created_at, i.sick, "
        "i.probability_sick, i.confidence_pct, i.confidence_level, i.damaged_pct, i.rust_level, "
        "i.doubtful_cut, i.model_version, i.response_ms, i.notes, i.status, i.reviewed_by, "
        "i.reviewed_at, i.true_label, i.review_note")
    i = db().one(f"SELECT {campos}, p.name AS plot_name, p.variety, f.name AS farm_name, f.id AS farm_id, "
                 "f.owner_id, a.username AS author_name, r.username AS reviewer_name "
                 "FROM inspections i JOIN plots p ON p.id = i.plot_id JOIN farms f ON f.id = p.farm_id "
                 "JOIN users a ON a.id = i.created_by LEFT JOIN users r ON r.id = i.reviewed_by "
                 "WHERE i.id = ?", (iid,))
    if i is None:
        abort(404)
    u = usuario()
    if not (es_personal(u) or i["owner_id"] == u["id"]):
        registrar("warning", "forbidden", f"inspection {iid}")
        abort(403)
    return i


def filtro_alcance(alias_finca="f"):
    """SQL fragment and arguments that limit rows to what the user may see."""
    u = usuario()
    if es_personal(u):
        return "1 = 1", []
    return f"{alias_finca}.owner_id = ?", [u["id"]]


def actualizar_estado(iid):
    """Workflow rule: waiting for review -> action required -> closed."""
    i = db().one("SELECT sick, confidence_level, doubtful_cut, reviewed_by FROM inspections WHERE id = ?", (iid,))
    abiertas = db().one("SELECT COUNT(*) AS n FROM recommendations WHERE inspection_id = ? "
                        "AND status IN ('pending', 'in_progress')", (iid,))["n"]
    necesita = bool(i["sick"]) or i["confidence_level"] == "low" or bool(i["doubtful_cut"])
    if necesita and i["reviewed_by"] is None:
        estado = "pending_review"
    elif abiertas:
        estado = "action_required"
    else:
        estado = "closed"
    db().run("UPDATE inspections SET status = ? WHERE id = ?", (estado, iid))
    return estado


# --------------------------------------------------------------- dashboard
@app.route("/dashboard")
@requiere_login
def panel():
    u = usuario()
    cond, args = filtro_alcance()
    base = ("FROM inspections i JOIN plots p ON p.id = i.plot_id JOIN farms f ON f.id = p.farm_id "
            f"WHERE {cond}")
    total = db().one(f"SELECT COUNT(*) AS n {base}", args)["n"]
    enfermas = db().one(f"SELECT COUNT(*) AS n {base} AND i.sick = 1", args)["n"]
    por_revisar = db().one(f"SELECT COUNT(*) AS n {base} AND i.status = 'pending_review'", args)["n"]
    acciones = db().one("SELECT COUNT(*) AS n FROM recommendations r JOIN inspections i ON i.id = r.inspection_id "
                        "JOIN plots p ON p.id = i.plot_id JOIN farms f ON f.id = p.farm_id "
                        f"WHERE {cond} AND r.status IN ('pending', 'in_progress')", args)["n"]
    fincas = db().one(f"SELECT COUNT(*) AS n FROM farms f WHERE {cond}", args)["n"]
    lotes = db().one("SELECT COUNT(*) AS n FROM plots p JOIN farms f ON f.id = p.farm_id "
                     f"WHERE {cond}", args)["n"]
    recientes = db().all("SELECT i.id, i.created_at, i.sick, i.confidence_pct, i.status, p.name AS plot_name, "
                         f"f.name AS farm_name {base} ORDER BY i.id DESC LIMIT 6", args)
    pendientes = db().all("SELECT r.id, r.action, r.status, r.variety_type, i.id AS inspection_id, "
                          "p.name AS plot_name, f.name AS farm_name "
                          "FROM recommendations r JOIN inspections i ON i.id = r.inspection_id "
                          "JOIN plots p ON p.id = i.plot_id JOIN farms f ON f.id = p.farm_id "
                          f"WHERE {cond} AND r.status IN ('pending', 'in_progress') "
                          "ORDER BY r.id DESC LIMIT 6", args)
    extra = {}
    if u["role"] == "admin":
        extra["usuarios"] = {r["role"]: r["n"] for r in db().all("SELECT role, COUNT(*) AS n FROM users GROUP BY role")}
    if es_personal(u):
        extra["feedback_abierto"] = db().one("SELECT COUNT(*) AS n FROM feedback WHERE status = 'open'")["n"]
    return render_template("dashboard.html", total=total, enfermas=enfermas, por_revisar=por_revisar,
                           acciones=acciones, fincas=fincas, lotes=lotes, recientes=recientes,
                           pendientes=pendientes, extra=extra)


# ------------------------------------------------------------------ farms
def validar_finca(form):
    nombre = form.get("nombre", "").strip()
    municipio = form.get("municipio", "").strip()
    depto = form.get("departamento", "").strip()
    if not (2 <= len(nombre) <= 80):
        return None, "The farm name must have 2 to 80 characters."
    if not (2 <= len(municipio) <= 80) or not (2 <= len(depto) <= 80):
        return None, "Municipality and department are required (2 to 80 characters)."
    return (nombre, municipio, depto), None


@app.route("/farms")
@requiere_login
def fincas():
    cond, args = filtro_alcance()
    filas = db().all("SELECT f.*, u.username AS owner_name FROM farms f JOIN users u ON u.id = f.owner_id "
                     f"WHERE {cond} ORDER BY f.name", args)
    conteo = {r["farm_id"]: r["n"] for r in db().all("SELECT farm_id, COUNT(*) AS n FROM plots GROUP BY farm_id")}
    return render_template("farms.html", fincas=filas, conteo=conteo)


@app.route("/farms/new", methods=["GET", "POST"])
@requiere_rol("grower", "admin")
def finca_nueva():
    u = usuario()
    duenos = db().all("SELECT id, username FROM users WHERE role = 'grower' AND active = 1 ORDER BY username") \
        if u["role"] == "admin" else []
    if request.method == "POST":
        datos, error = validar_finca(request.form)
        dueno = u["id"]
        if u["role"] == "admin":
            try:
                dueno = int(request.form.get("owner_id", ""))
            except ValueError:
                error = error or "Choose the owner of the farm."
            else:
                if not db().one("SELECT id FROM users WHERE id = ? AND role = 'grower' AND active = 1", (dueno,)):
                    error = error or "The owner must be an active coffee grower."
        if not error and db().one("SELECT id FROM farms WHERE owner_id = ? AND name = ?", (dueno, datos[0])):
            error = "That owner already has a farm with that name."
        if error:
            return render_template("farm_form.html",
                                   finca={"name": request.form.get("nombre", ""),
                                          "municipality": request.form.get("municipio", ""),
                                          "department": request.form.get("departamento", ""), "owner_id": dueno},
                                   duenos=duenos, error=error, nuevo=True), 400
        fid = db().insert("INSERT INTO farms (name, municipality, department, owner_id, created_at) "
                          "VALUES (?, ?, ?, ?, ?)", (*datos, dueno, now()))
        registrar("info", "farm_created", f"farm {fid}")
        flash("Farm created. Now add its plots.")
        return redirect(url_for("finca", fid=fid))
    return render_template("farm_form.html", finca={}, duenos=duenos, error=None, nuevo=True)


@app.route("/farms/<int:fid>")
@requiere_login
def finca(fid):
    f = finca_o_error(fid)
    lotes = db().all("SELECT p.*, (SELECT COUNT(*) FROM inspections i WHERE i.plot_id = p.id) AS n_insp, "
                     "(SELECT MAX(created_at) FROM inspections i WHERE i.plot_id = p.id) AS ultima "
                     "FROM plots p WHERE p.farm_id = ? ORDER BY p.name", (fid,))
    return render_template("farm.html", f=f, lotes=lotes, puede_editar=puede_editar_finca(f))


def puede_editar_finca(f):
    u = usuario()
    return u["role"] == "admin" or (u["role"] == "grower" and f["owner_id"] == u["id"])


@app.route("/farms/<int:fid>/edit", methods=["GET", "POST"])
@requiere_rol("grower", "admin")
def finca_editar(fid):
    f = finca_o_error(fid)
    if not puede_editar_finca(f):
        abort(403)
    if request.method == "POST":
        datos, error = validar_finca(request.form)
        if not error and db().one("SELECT id FROM farms WHERE owner_id = ? AND name = ? AND id <> ?",
                                  (f["owner_id"], datos[0], fid)):
            error = "That owner already has a farm with that name."
        if error:
            return render_template("farm_form.html", finca={"id": fid, "name": request.form.get("nombre", ""),
                                   "municipality": request.form.get("municipio", ""),
                                   "department": request.form.get("departamento", ""),
                                   "owner_id": f["owner_id"]}, duenos=[], error=error, nuevo=False), 400
        db().run("UPDATE farms SET name = ?, municipality = ?, department = ? WHERE id = ?", (*datos, fid))
        registrar("info", "farm_updated", f"farm {fid}")
        flash("Farm updated.")
        return redirect(url_for("finca", fid=fid))
    return render_template("farm_form.html", finca=f, duenos=[], error=None, nuevo=False)


@app.route("/farms/<int:fid>/delete", methods=["POST"])
@requiere_rol("grower", "admin")
def finca_borrar(fid):
    f = finca_o_error(fid)
    if not puede_editar_finca(f):
        abort(403)
    # Business rule: a farm with plots cannot be deleted
    if db().one("SELECT COUNT(*) AS n FROM plots WHERE farm_id = ?", (fid,))["n"]:
        flash("This farm still has plots. Delete the plots first (a plot with inspections cannot be deleted).")
        return redirect(url_for("finca", fid=fid))
    db().run("DELETE FROM farms WHERE id = ?", (fid,))
    registrar("info", "farm_deleted", f"farm {fid}")
    flash("Farm deleted.")
    return redirect(url_for("fincas"))


# ------------------------------------------------------------------ plots
def validar_lote(form):
    nombre = form.get("nombre", "").strip()
    variedad = form.get("variedad", "")
    notas = form.get("notas", "").strip()[:500]
    if not (1 <= len(nombre) <= 80):
        return None, "The plot name must have 1 to 80 characters."
    if variedad not in VARIETIES:
        return None, "Choose the variety type."
    try:
        area = float(form.get("area", "").replace(",", "."))
    except ValueError:
        return None, "The area must be a number (hectares)."
    if not (0 < area <= 10000):
        return None, "The area must be greater than 0 and at most 10,000 hectares."
    anio = form.get("anio", "").strip()
    if anio:
        if not anio.isdigit() or not (1950 <= int(anio) <= datetime.now().year):
            return None, "The planting year must be between 1950 and this year."
        anio = int(anio)
    else:
        anio = None
    return (nombre, variedad, area, anio, notas), None


@app.route("/farms/<int:fid>/plots/new", methods=["GET", "POST"])
@requiere_rol("grower", "admin")
def lote_nuevo(fid):
    f = finca_o_error(fid)
    if not puede_editar_finca(f):
        abort(403)
    if request.method == "POST":
        datos, error = validar_lote(request.form)
        if not error and db().one("SELECT id FROM plots WHERE farm_id = ? AND name = ?", (fid, datos[0])):
            error = "This farm already has a plot with that name."
        if error:
            return render_template("plot_form.html", f=f, lote=request.form, error=error, nuevo=True), 400
        pid = db().insert("INSERT INTO plots (farm_id, name, variety, area_ha, planted_year, notes, created_at) "
                          "VALUES (?, ?, ?, ?, ?, ?, ?)", (fid, *datos, now()))
        registrar("info", "plot_created", f"plot {pid}")
        flash("Plot created.")
        return redirect(url_for("lote", pid=pid))
    return render_template("plot_form.html", f=f, lote={}, error=None, nuevo=True)


@app.route("/plots/<int:pid>")
@requiere_login
def lote(pid):
    p = lote_o_error(pid)
    inspecciones = db().all("SELECT i.id, i.created_at, i.sick, i.confidence_pct, i.confidence_level, "
                            "i.damaged_pct, i.status, a.username AS author_name FROM inspections i "
                            "JOIN users a ON a.id = i.created_by WHERE i.plot_id = ? ORDER BY i.id DESC", (pid,))
    return render_template("plot.html", p=p, inspecciones=inspecciones,
                           puede_editar=puede_editar_finca({"owner_id": p["owner_id"]}))


@app.route("/plots/<int:pid>/edit", methods=["GET", "POST"])
@requiere_rol("grower", "admin")
def lote_editar(pid):
    p = lote_o_error(pid)
    if not puede_editar_finca({"owner_id": p["owner_id"]}):
        abort(403)
    f = {"id": p["farm_id"], "name": p["farm_name"]}
    if request.method == "POST":
        datos, error = validar_lote(request.form)
        if not error and db().one("SELECT id FROM plots WHERE farm_id = ? AND name = ? AND id <> ?",
                                  (p["farm_id"], datos[0], pid)):
            error = "This farm already has a plot with that name."
        if error:
            return render_template("plot_form.html", f=f, lote=request.form, error=error, nuevo=False, pid=pid), 400
        db().run("UPDATE plots SET name = ?, variety = ?, area_ha = ?, planted_year = ?, notes = ? WHERE id = ?",
                 (*datos, pid))
        registrar("info", "plot_updated", f"plot {pid}")
        flash("Plot updated.")
        return redirect(url_for("lote", pid=pid))
    valores = {"nombre": p["name"], "variedad": p["variety"], "area": p["area_ha"],
               "anio": p["planted_year"] or "", "notas": p["notes"]}
    return render_template("plot_form.html", f=f, lote=valores, error=None, nuevo=False, pid=pid)


@app.route("/plots/<int:pid>/delete", methods=["POST"])
@requiere_rol("grower", "admin")
def lote_borrar(pid):
    p = lote_o_error(pid)
    if not puede_editar_finca({"owner_id": p["owner_id"]}):
        abort(403)
    # Business rule: inspections are the history of the model, so a plot that has them is kept
    if db().one("SELECT COUNT(*) AS n FROM inspections WHERE plot_id = ?", (pid,))["n"]:
        flash("This plot has inspections, so it cannot be deleted. Its history must be kept.")
        return redirect(url_for("lote", pid=pid))
    db().run("DELETE FROM plots WHERE id = ?", (pid,))
    registrar("info", "plot_deleted", f"plot {pid}")
    flash("Plot deleted.")
    return redirect(url_for("finca", fid=p["farm_id"]))


# -------------------------------------------------------------- inspections
@app.route("/plots/<int:pid>/inspect", methods=["GET", "POST"])
@requiere_rol("grower", "technician", "admin")
def inspeccionar(pid):
    p = lote_o_error(pid)
    u = usuario()
    if request.method == "GET":
        return render_template("inspect.html", p=p, error=None)
    archivo = request.files.get("foto")
    if archivo is None or archivo.filename == "":
        return render_template("inspect.html", p=p, error="Choose a photo of a leaf."), 400
    hace_una_hora = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
    if db().one("SELECT COUNT(*) AS n FROM inspections WHERE created_by = ? AND created_at > ?",
                (u["id"], hace_una_hora))["n"] >= LIMITE_FOTOS_POR_HORA:
        return render_template("inspect.html", p=p, error="You reached the limit of "
                               f"{LIMITE_FOTOS_POR_HORA} photos per hour. Try again later."), 429
    try:
        r = analizar(archivo, p["variety"])
    except ValueError as e:
        registrar("warning", "unreadable_photo", f"plot {pid}")
        return render_template("inspect.html", p=p, error=str(e)), 400
    except Exception as e:
        registrar("error", "inference_error", f"plot {pid}: {e!r}")
        return render_template("inspect.html", p=p, error="The analysis failed. It was logged. Try another photo."), 500
    iid = db().insert(
        "INSERT INTO inspections (plot_id, created_by, created_at, sick, probability_sick, confidence_pct, "
        "confidence_level, damaged_pct, rust_level, doubtful_cut, model_version, response_ms, image_b64, "
        "features, notes, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending_review')",
        (pid, u["id"], now(), int(r["enferma"]), r["probabilidad"], r["porcentaje"], r["nivel"], r["afectada"],
         r["nivel_roya"], int(r["dudosa"]), MODEL_VERSION, r["ms"], r["imagen"], json.dumps(r["valores"]),
         request.form.get("notas", "").strip()[:500]))
    # The model output becomes actions in the business process
    for rec in r["recomendaciones"]:
        db().insert("INSERT INTO recommendations (inspection_id, variety_type, action, text, status, note, "
                    "created_at, updated_at) VALUES (?, ?, ?, ?, 'pending', '', ?, ?)",
                    (iid, rec["variedad"], rec["accion"], rec["texto"], now(), now()))
    actualizar_estado(iid)
    registrar("info", "inference", f"inspection {iid} sick={int(r['enferma'])} conf={r['porcentaje']}", r["ms"])
    return redirect(url_for("inspeccion", iid=iid))


@app.route("/inspections")
@requiere_login
def inspecciones():
    cond, args = filtro_alcance()
    estado = request.args.get("status", "")
    if estado in ESTADOS:
        cond += " AND i.status = ?"
        args.append(estado)
    resultado = request.args.get("result", "")
    if resultado in ("sick", "healthy"):
        cond += " AND i.sick = ?"
        args.append(1 if resultado == "sick" else 0)
    filas = db().all("SELECT i.id, i.created_at, i.sick, i.confidence_pct, i.confidence_level, i.damaged_pct, "
                     "i.status, i.true_label, i.model_version, p.name AS plot_name, f.name AS farm_name, "
                     "a.username AS author_name FROM inspections i JOIN plots p ON p.id = i.plot_id "
                     "JOIN farms f ON f.id = p.farm_id JOIN users a ON a.id = i.created_by "
                     f"WHERE {cond} ORDER BY i.id DESC LIMIT 200", args)
    return render_template("inspections.html", filas=filas, estado=estado, resultado=resultado, ESTADOS=ESTADOS)


@app.route("/inspections/<int:iid>")
@requiere_login
def inspeccion(iid):
    i = inspeccion_o_error(iid)
    recs = db().all("SELECT r.*, u.username AS updater FROM recommendations r "
                    "LEFT JOIN users u ON u.id = r.updated_by WHERE r.inspection_id = ? ORDER BY r.id", (iid,))
    return render_template("inspection.html", i=i, recs=recs, ESTADOS_ACCION=ESTADOS_ACCION)


@app.route("/inspections/<int:iid>/review", methods=["POST"])
@requiere_rol("technician", "admin")
def revisar(iid):
    i = inspeccion_o_error(iid, con_imagen=False)
    etiqueta = request.form.get("true_label", "")
    nota = request.form.get("nota", "").strip()[:500]
    if etiqueta not in ("healthy", "sick", "unsure"):
        flash("Choose what you found on the leaf: healthy, sick or not sure.")
        return redirect(url_for("inspeccion", iid=iid))
    u = usuario()
    db().run("UPDATE inspections SET reviewed_by = ?, reviewed_at = ?, true_label = ?, review_note = ? "
             "WHERE id = ?", (u["id"], now(), etiqueta, nota, iid))
    # Business rules that depend on the technician's finding
    if etiqueta == "healthy" and i["sick"]:
        db().run("UPDATE recommendations SET status = 'dismissed', note = ?, updated_by = ?, updated_at = ? "
                 "WHERE inspection_id = ? AND status IN ('pending', 'in_progress')",
                 ("Dismissed: the technician found the leaf healthy.", u["id"], now(), iid))
    if etiqueta == "sick" and not i["sick"]:
        db().insert("INSERT INTO recommendations (inspection_id, variety_type, action, text, status, note, "
                    "created_at, updated_at) VALUES (?, ?, 'Technician follow-up', ?, 'pending', '', ?, ?)",
                    (iid, "unknown", "The model said healthy, but the technician found signs of disease. "
                     "Follow the technician's note and check the nearby plants.", now(), now()))
    actualizar_estado(iid)
    registrar("info", "review", f"inspection {iid} label={etiqueta} model_sick={i['sick']}")
    flash("Review saved.")
    return redirect(url_for("inspeccion", iid=iid))


@app.route("/inspections/<int:iid>/delete", methods=["POST"])
@requiere_rol("admin")
def inspeccion_borrar(iid):
    inspeccion_o_error(iid, con_imagen=False)
    db().run("DELETE FROM feedback WHERE inspection_id = ?", (iid,))
    db().run("DELETE FROM recommendations WHERE inspection_id = ?", (iid,))
    db().run("DELETE FROM inspections WHERE id = ?", (iid,))
    registrar("warning", "inspection_deleted", f"inspection {iid}")
    flash("Inspection deleted.")
    return redirect(url_for("inspecciones"))


@app.route("/recommendations/<int:rid>/update", methods=["POST"])
@requiere_login
def actualizar_accion(rid):
    r = db().one("SELECT * FROM recommendations WHERE id = ?", (rid,))
    if r is None:
        abort(404)
    i = inspeccion_o_error(r["inspection_id"], con_imagen=False)
    u = usuario()
    nuevo = request.form.get("status", "")
    nota = request.form.get("nota", "").strip()[:500]
    if nuevo not in ("in_progress", "done", "dismissed"):
        abort(400, "Unknown status.")
    if nuevo == "dismissed" and not es_personal(u):
        flash("Only a technician or an administrator can dismiss an action.")
        return redirect(url_for("inspeccion", iid=i["id"]))
    # Business rule: a fungicide is applied only after a technician reviewed the inspection
    if nuevo == "done" and r["action"] == "Apply fungicide" and i["reviewed_by"] is None:
        flash("A technician must review this inspection before the fungicide is marked as done.")
        return redirect(url_for("inspeccion", iid=i["id"]))
    db().run("UPDATE recommendations SET status = ?, note = ?, updated_by = ?, updated_at = ? WHERE id = ?",
             (nuevo, nota or r["note"], u["id"], now(), rid))
    actualizar_estado(i["id"])
    registrar("info", "action_update", f"recommendation {rid} -> {nuevo}")
    flash("Action updated.")
    return redirect(url_for("inspeccion", iid=i["id"]))


# --------------------------------------------------------------- feedback
@app.route("/feedback")
@requiere_login
def comentarios():
    u = usuario()
    if es_personal(u):
        filas = db().all("SELECT fb.*, u.username FROM feedback fb JOIN users u ON u.id = fb.user_id "
                         "ORDER BY fb.id DESC LIMIT 200")
    else:
        filas = db().all("SELECT fb.*, u.username FROM feedback fb JOIN users u ON u.id = fb.user_id "
                         "WHERE fb.user_id = ? ORDER BY fb.id DESC LIMIT 200", (u["id"],))
    return render_template("feedback.html", filas=filas)


@app.route("/feedback/new", methods=["GET", "POST"])
@requiere_login
def comentario_nuevo():
    u = usuario()
    iid = request.values.get("inspection", type=int)
    if iid:
        inspeccion_o_error(iid, con_imagen=False)      # same access rule as the inspection itself
    if request.method == "POST":
        tipo = request.form.get("kind", "feedback")
        mensaje = request.form.get("mensaje", "").strip()[:1000]
        correcta = request.form.get("correct", "")
        if tipo not in ("feedback", "incident"):
            tipo = "feedback"
        if len(mensaje) < 3 and correcta == "":
            return render_template("feedback_form.html", iid=iid, error="Write a short message or answer "
                                   "whether the result was right.", mensaje=mensaje), 400
        db().insert("INSERT INTO feedback (user_id, inspection_id, kind, answer_correct, message, status, "
                    "response, created_at) VALUES (?, ?, ?, ?, ?, 'open', '', ?)",
                    (u["id"], iid, tipo, None if correcta == "" else int(correcta == "1"), mensaje, now()))
        registrar("info", "feedback", f"{tipo} inspection={iid}")
        flash("Thank you. Your message was saved.")
        return redirect(url_for("inspeccion", iid=iid) if iid else url_for("comentarios"))
    return render_template("feedback_form.html", iid=iid, error=None, mensaje="")


@app.route("/feedback/<int:cid>/resolve", methods=["POST"])
@requiere_rol("technician", "admin")
def comentario_resolver(cid):
    if db().one("SELECT id FROM feedback WHERE id = ?", (cid,)) is None:
        abort(404)
    db().run("UPDATE feedback SET status = 'resolved', response = ?, resolved_by = ? WHERE id = ?",
             (request.form.get("respuesta", "").strip()[:500], usuario()["id"], cid))
    registrar("info", "feedback_resolved", f"feedback {cid}")
    flash("Marked as resolved.")
    return redirect(url_for("comentarios"))


# ------------------------------------------------------------------ admin
@app.route("/admin/users")
@requiere_rol("admin")
def admin_usuarios():
    filas = db().all("SELECT id, username, full_name, role, active, created_at, last_login FROM users ORDER BY id")
    return render_template("admin_users.html", filas=filas)


@app.route("/admin/users/new", methods=["GET", "POST"])
@requiere_rol("admin")
def admin_usuario_nuevo():
    if request.method == "POST":
        nombre = request.form.get("usuario", "").strip()
        completo = request.form.get("nombre", "").strip()[:80]
        clave = request.form.get("clave", "")
        rol = request.form.get("rol", "")
        if not USUARIO_VALIDO.match(nombre):
            error = "The username must have 3 to 30 characters: letters, numbers or _."
        elif rol not in ROLES:
            error = "Choose a role."
        elif len(clave) < 8 or not re.search(r"[A-Za-z]", clave) or not re.search(r"\d", clave):
            error = "The password must have at least 8 characters, with letters and numbers."
        elif db().one("SELECT id FROM users WHERE username_lc = ?", (nombre.lower(),)):
            error = "That username is already taken."
        else:
            nuevo = db().insert("INSERT INTO users (username, username_lc, full_name, password_hash, role, "
                                "active, created_at) VALUES (?, ?, ?, ?, ?, 1, ?)",
                                (nombre, nombre.lower(), completo, generate_password_hash(clave), rol, now()))
            registrar("warning", "user_created", f"user {nuevo} role={rol}")
            flash("Account created.")
            return redirect(url_for("admin_usuarios"))
        return render_template("admin_user_form.html", error=error, nombre=nombre, completo=completo), 400
    return render_template("admin_user_form.html", error=None, nombre="", completo="")


@app.route("/admin/users/<int:uid>/update", methods=["POST"])
@requiere_rol("admin")
def admin_usuario_actualizar(uid):
    objetivo = db().one("SELECT id, username, role, active FROM users WHERE id = ?", (uid,))
    if objetivo is None:
        abort(404)
    yo = usuario()
    rol = request.form.get("rol", objetivo["role"])
    activo = 1 if request.form.get("activo") == "1" else 0
    if rol not in ROLES:
        abort(400, "Unknown role.")
    # Rule: the system must always keep one active administrator, and you cannot lock yourself out
    if objetivo["id"] == yo["id"] and (rol != "admin" or not activo):
        flash("You cannot remove your own administrator access or disable your own account.")
        return redirect(url_for("admin_usuarios"))
    if objetivo["role"] == "admin" and (rol != "admin" or not activo):
        otros = db().one("SELECT COUNT(*) AS n FROM users WHERE role = 'admin' AND active = 1 AND id <> ?", (uid,))["n"]
        if otros == 0:
            flash("There must be at least one active administrator.")
            return redirect(url_for("admin_usuarios"))
    db().run("UPDATE users SET role = ?, active = ? WHERE id = ?", (rol, activo, uid))
    registrar("warning", "user_updated", f"user {uid} role={rol} active={activo}")
    flash("User updated.")
    return redirect(url_for("admin_usuarios"))


@app.route("/admin/logs")
@requiere_rol("admin")
def admin_registro():
    nivel = request.args.get("level", "")
    if nivel in ("info", "warning", "error"):
        filas = db().all("SELECT * FROM event_log WHERE level = ? ORDER BY id DESC LIMIT 300", (nivel,))
    else:
        filas = db().all("SELECT * FROM event_log ORDER BY id DESC LIMIT 300")
    return render_template("admin_logs.html", filas=filas, nivel=nivel)


# ------------------------------------------------------ reports and monitoring
@app.route("/reports")
@requiere_login
def informes():
    cond, args = filtro_alcance()
    filas = db().all("SELECT p.id AS plot_id, p.name AS plot_name, p.variety, f.id AS farm_id, f.name AS farm_name, "
                     "(SELECT COUNT(*) FROM inspections i WHERE i.plot_id = p.id) AS n, "
                     "(SELECT COUNT(*) FROM inspections i WHERE i.plot_id = p.id AND i.sick = 1) AS n_sick, "
                     "(SELECT MAX(created_at) FROM inspections i WHERE i.plot_id = p.id) AS ultima, "
                     "(SELECT COUNT(*) FROM recommendations r JOIN inspections i ON i.id = r.inspection_id "
                     "WHERE i.plot_id = p.id AND r.status IN ('pending', 'in_progress')) AS abiertas "
                     f"FROM plots p JOIN farms f ON f.id = p.farm_id WHERE {cond} ORDER BY f.name, p.name", args)
    return render_template("reports.html", filas=filas)


@app.route("/reports/inspections.csv")
@requiere_login
def informe_csv():
    cond, args = filtro_alcance()
    filas = db().all("SELECT i.id, i.created_at, f.name AS farm, p.name AS plot, a.username AS user_name, i.sick, "
                     "i.confidence_pct, i.damaged_pct, i.rust_level, i.model_version, i.response_ms, i.status, "
                     "i.true_label FROM inspections i JOIN plots p ON p.id = i.plot_id "
                     "JOIN farms f ON f.id = p.farm_id JOIN users a ON a.id = i.created_by "
                     f"WHERE {cond} ORDER BY i.id", args)
    salida = io.StringIO()
    escritor = csv.writer(salida)
    escritor.writerow(["id", "date_utc", "farm", "plot", "user", "model_says_sick", "confidence_pct",
                       "damaged_pct", "rust_level_estimate", "model_version", "response_ms", "status",
                       "technician_label"])
    for r in filas:
        escritor.writerow([r["id"], r["created_at"], r["farm"], r["plot"], r["user_name"], r["sick"],
                           r["confidence_pct"], r["damaged_pct"], r["rust_level"], r["model_version"],
                           r["response_ms"], r["status"], r["true_label"] or ""])
    registrar("info", "report_csv", f"{len(filas)} rows")
    return Response(salida.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=inspections.csv"})


def indicadores_monitoreo():
    """Numbers for the monitoring page: usage, speed, errors, model quality and data drift."""
    hace7 = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
    insp = db().all("SELECT sick, confidence_level, doubtful_cut, response_ms, true_label, created_at, features, "
                    "model_version FROM inspections ORDER BY id")
    ind = {"total": len(insp), "ultimos7": sum(1 for r in insp if r["created_at"] >= hace7)}
    tiempos = sorted(r["response_ms"] for r in insp)
    ind["ms_media"] = round(sum(tiempos) / len(tiempos)) if tiempos else None
    ind["ms_p95"] = tiempos[min(len(tiempos) - 1, int(0.95 * len(tiempos)))] if tiempos else None
    ind["pct_baja"] = round(100 * sum(r["confidence_level"] == "low" for r in insp) / len(insp), 1) if insp else None
    ind["pct_dudosa"] = round(100 * sum(bool(r["doubtful_cut"]) for r in insp) / len(insp), 1) if insp else None
    ind["pct_enferma"] = round(100 * sum(bool(r["sick"]) for r in insp) / len(insp), 1) if insp else None
    ind["errores7"] = db().one("SELECT COUNT(*) AS n FROM event_log WHERE level = 'error' AND ts >= ?", (hace7,))["n"]
    ind["fallos_login7"] = db().one("SELECT COUNT(*) AS n FROM event_log WHERE event = 'login_failed' AND ts >= ?",
                                    (hace7,))["n"]
    # model quality against what technicians found
    rev = [r for r in insp if r["true_label"] in ("healthy", "sick")]
    ind["n_revisadas"] = len(rev)
    if rev:
        aciertos = sum((r["true_label"] == "sick") == bool(r["sick"]) for r in rev)
        ind["exactitud"] = round(100 * aciertos / len(rev), 1)
        ind["fp"] = sum(bool(r["sick"]) and r["true_label"] == "healthy" for r in rev)
        ind["fn"] = sum((not r["sick"]) and r["true_label"] == "sick" for r in rev)
    else:
        ind["exactitud"] = ind["fp"] = ind["fn"] = None
    fb = db().all("SELECT answer_correct FROM feedback WHERE answer_correct IS NOT NULL")
    ind["n_fb"] = len(fb)
    ind["pct_fb_correcta"] = round(100 * sum(bool(r["answer_correct"]) for r in fb) / len(fb), 1) if fb else None
    ind["abiertos"] = db().one("SELECT COUNT(*) AS n FROM feedback WHERE status = 'open'")["n"]
    ind["versiones"] = sorted({r["model_version"] for r in insp})
    # data drift: mean of the 26 features of the last inspections against the training reference
    deriva, minimo = [], 20
    ultimas = insp[-200:]
    if REFERENCIA and len(ultimas) >= minimo:
        matriz = np.array([json.loads(r["features"]) for r in ultimas], dtype=float)
        if matriz.shape[1] == len(car.NOMBRES):
            for k, nombre in enumerate(car.NOMBRES):
                mu, sd = REFERENCIA["media"].get(nombre), REFERENCIA["desv"].get(nombre)
                if mu is None or not sd:
                    continue
                z = (float(matriz[:, k].mean()) - mu) / sd
                deriva.append({"nombre": nombre, "media_ref": round(mu, 3), "media_nueva": round(float(matriz[:, k].mean()), 3),
                               "z": round(z, 2), "alarma": abs(z) > 0.5})
            deriva.sort(key=lambda d: -abs(d["z"]))
    ind["deriva"] = deriva
    ind["deriva_n"] = len(ultimas)
    ind["deriva_minimo"] = minimo
    ind["deriva_alarmas"] = sum(d["alarma"] for d in deriva)
    return ind


@app.route("/monitoring")
@requiere_rol("admin", "technician")
def monitoreo():
    return render_template("monitoring.html", ind=indicadores_monitoreo())


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
