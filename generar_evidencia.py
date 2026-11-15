# Paso 11 - Evidencia para el informe y las fases del proyecto (CRISP-ML)
# Genera tablas, gráficas y un resumen con números reales. Todo queda en la carpeta evidencia/.
# Usa la misma división por plantas (80/20, semilla 42) que entrenar_modelo_app.py.
# La prueba (20%) NO se usa para elegir hiperparámetros: se elige solo con validación cruzada en el 80%.

# 1. Importar librerías necesarias
import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, mean_absolute_error,
                             precision_recall_fscore_support, r2_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import caracteristicas as car

warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parent
OUT = RAIZ / "evidencia"
MODELOS = RAIZ / "models"
OUT.mkdir(exist_ok=True)
MODELOS.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")
resumen = {}


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(OUT / nombre, dpi=150)
    plt.close(fig)


# 2. Carga de datos (igual que entrenar_modelo_app.py)
exp = pd.read_csv(RAIZ / "data.csv")                       # contorno del experto
etiquetas = exp[["archivo", "clase", "target"]]
data = pd.read_csv(RAIZ / "data_grabcut.csv").merge(etiquetas, on="archivo")   # corte automático
data["planta"] = data["archivo"].str.extract(r"^(C\d+P\d+)")[0]
exp = data[["archivo", "planta"]].merge(exp, on="archivo", how="left")   # mismo orden que data
X, y = data[car.NOMBRES], data["target"]

# 3. División por planta, igual que la app
division = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
i_tr, i_te = next(division.split(X, y, groups=data["planta"]))
X_tr, X_te, y_tr, y_te = X.iloc[i_tr], X.iloc[i_te], y.iloc[i_tr], y.iloc[i_te]
g_tr = data["planta"].iloc[i_tr]
plantas_comunes = set(data["planta"].iloc[i_tr]) & set(data["planta"].iloc[i_te])
print(f"Fotos: {len(data)} | entrenamiento {len(X_tr)} | prueba {len(X_te)}")
print(f"Plantas en entrenamiento: {g_tr.nunique()} | en prueba: {data['planta'].iloc[i_te].nunique()} "
      f"| plantas en los dos: {len(plantas_comunes)}")
resumen["division"] = {"fotos": len(data), "train": len(X_tr), "test": len(X_te),
                       "plantas_train": int(g_tr.nunique()),
                       "plantas_test": int(data["planta"].iloc[i_te].nunique()),
                       "plantas_en_ambos": len(plantas_comunes)}

# ====================================================== FASE 2: CALIDAD Y PROCESAMIENTO
fotos_planta = data.groupby("planta").size()
calidad = {
    "filas": len(data),
    "columnas_numericas": len(car.NOMBRES),
    "valores_nulos": int(data[car.NOMBRES].isna().sum().sum()),
    "valores_infinitos": int(np.isinf(data[car.NOMBRES]).sum().sum()),
    "fotos_duplicadas_por_nombre": int(data["archivo"].duplicated().sum()),
    "filas_duplicadas_en_caracteristicas": int(data[car.NOMBRES].duplicated().sum()),
    "columnas_constantes": [c for c in car.NOMBRES if data[c].nunique() <= 1],
    "plantas": int(fotos_planta.size),
    "fotos_por_planta_min_max": [int(fotos_planta.min()), int(fotos_planta.max())],
    "nulos_en_data_experto": int(exp[car.NOMBRES].isna().sum().sum()),
}
print("\n=== FASE 2: calidad de los datos ===")
for k, v in calidad.items():
    print(f"  {k}: {v}")
resumen["calidad"] = calidad

# Valores atípicos (regla del rango intercuartil, 1.5 x IQR)
filas = []
marca = pd.DataFrame(False, index=data.index, columns=car.NOMBRES)
for c in car.NOMBRES:
    q1, q3 = data[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    marca[c] = (data[c] < q1 - 1.5 * iqr) | (data[c] > q3 + 1.5 * iqr)
    filas.append({"caracteristica": c, "atipicos": int(marca[c].sum()),
                  "porcentaje": round(marca[c].mean() * 100, 1)})
atip = pd.DataFrame(filas).sort_values("atipicos", ascending=False)
atip.to_csv(OUT / "atipicos.csv", index=False)
alguna = marca.any(axis=1)
enf_en_atipicos = data.loc[alguna, "target"].mean() * 100
enf_total = data["target"].mean() * 100
print(f"  Fotos con algún valor atípico: {alguna.sum()} ({alguna.mean() * 100:.1f}%)")
print(f"  De esas, enfermas: {enf_en_atipicos:.1f}% (en todo el conjunto: {enf_total:.1f}%)")
print("  Las 5 características con más atípicos:")
print(atip.head(5).to_string(index=False))
resumen["atipicos"] = {"fotos_con_atipicos": int(alguna.sum()),
                       "porcentaje": round(alguna.mean() * 100, 1),
                       "enfermas_entre_atipicos_pct": round(enf_en_atipicos, 1),
                       "enfermas_en_total_pct": round(enf_total, 1),
                       "top5": atip.head(5).to_dict("records")}

# Antes y después: estandarización ajustada SOLO con entrenamiento
esc = StandardScaler().fit(X_tr)
Xtr_e = pd.DataFrame(esc.transform(X_tr), columns=car.NOMBRES)
Xte_e = pd.DataFrame(esc.transform(X_te), columns=car.NOMBRES)
elegidas = car.NOMBRES[:6]
tabla = []
for c in elegidas:
    tabla.append({"caracteristica": c,
                  "antes_train_media": X_tr[c].mean(), "antes_train_desv": X_tr[c].std(),
                  "antes_test_media": X_te[c].mean(), "antes_test_desv": X_te[c].std(),
                  "despues_train_media": Xtr_e[c].mean(), "despues_train_desv": Xtr_e[c].std(),
                  "despues_test_media": Xte_e[c].mean(), "despues_test_desv": Xte_e[c].std()})
antes_despues = pd.DataFrame(tabla).round(3)
antes_despues.to_csv(OUT / "antes_despues.csv", index=False)
print("\n  Antes y después de estandarizar (media y desviación; el ajuste usa solo entrenamiento):")
print(antes_despues.to_string(index=False))
resumen["antes_despues"] = antes_despues.to_dict("records")

fig, ejes = plt.subplots(2, 3, figsize=(11, 5.5))
for k, c in enumerate(elegidas[:3]):
    sns.histplot(X_tr[c], bins=30, ax=ejes[0, k], color="#A5531A")
    ejes[0, k].set_title(f"Before: {c}")
    ejes[0, k].set_xlabel("")
    sns.histplot(Xtr_e[c], bins=30, ax=ejes[1, k], color="#2F6B47")
    ejes[1, k].set_title(f"After (standardized): {c}")
    ejes[1, k].set_xlabel("")
guardar(fig, "fig_antes_despues.png")

orden = ["healthy", "red_spider_mite", "rust_level_1", "rust_level_2", "rust_level_3", "rust_level_4"]
cuenta = data["clase"].value_counts().reindex(orden)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [3, 1.3]})
a1.barh(cuenta.index[::-1], cuenta.values[::-1], color="#2F6B47")
a1.set_title("Before: 6 classes")
for i, v in enumerate(cuenta.values[::-1]):
    a1.text(v + 8, i, str(v), va="center")
b = data["target"].map({0: "Healthy", 1: "Sick"}).value_counts().reindex(["Healthy", "Sick"])
a2.bar(b.index, b.values, color=["#2F6B47", "#A5531A"])
a2.set_title("After: 2 classes")
for i, v in enumerate(b.values):
    a2.text(i, v + 10, str(v), ha="center")
guardar(fig, "fig_clases.png")
resumen["clases"] = {k: int(v) for k, v in cuenta.items()}
resumen["binaria"] = {"sana": int(b["Healthy"]), "enferma": int(b["Sick"])}

# ====================================================== FASE 3: EXPERIMENTOS
cv = GroupKFold(n_splits=5)
experimentos = []


def correr(familia, nombre_param, params, modelo):
    t0 = time.time()
    p = cross_val_score(modelo, X_tr, y_tr, groups=g_tr, cv=cv, scoring="accuracy", n_jobs=1)
    experimentos.append({"familia": familia, "parametros": nombre_param, "cv_media": p.mean() * 100,
                         "cv_desv": p.std() * 100, "segundos": round(time.time() - t0, 1),
                         "_modelo": modelo})
    print(f"  {familia:20s} {nombre_param:42s} {p.mean() * 100:6.2f}% (±{p.std() * 100:.2f})")


print("\n=== FASE 3: experimentos (validación cruzada por planta, solo con entrenamiento) ===")
for C in [0.01, 0.1, 1, 10, 100]:
    correr("Logistic Regression", f"C={C}", {"C": C},
           make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)))
for k in [3, 5, 7, 11, 15, 25]:
    correr("KNN", f"k={k}", {"k": k}, make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)))
for n in [100, 300]:
    for prof in [None, 6, 12]:
        for hoja in [1, 3]:
            correr("Random Forest", f"trees={n}, max_depth={prof}, min_leaf={hoja}", {},
                   RandomForestClassifier(n_estimators=n, max_depth=prof, min_samples_leaf=hoja,
                                          random_state=42, n_jobs=-1))
for lr in [0.05, 0.1]:
    for prof in [None, 4]:
        correr("Gradient Boosting", f"learning_rate={lr}, max_depth={prof}", {},
               HistGradientBoostingClassifier(learning_rate=lr, max_depth=prof, random_state=42))

tabla_exp = pd.DataFrame([{k: v for k, v in e.items() if k != "_modelo"} for e in experimentos])
tabla_exp.insert(0, "id", [f"E{i:02d}" for i in range(1, len(tabla_exp) + 1)])
tabla_exp["cv_media"] = tabla_exp["cv_media"].round(2)
tabla_exp["cv_desv"] = tabla_exp["cv_desv"].round(2)
tabla_exp.to_csv(OUT / "experimentos.csv", index=False)
resumen["n_experimentos"] = len(tabla_exp)

mejores = {}
for fam in tabla_exp["familia"].unique():
    idx = tabla_exp[tabla_exp["familia"] == fam]["cv_media"].idxmax()
    mejores[fam] = (tabla_exp.loc[idx], experimentos[idx]["_modelo"])
print("\n  Mejor configuración de cada familia (por validación cruzada):")
for fam, (fila, _) in mejores.items():
    print(f"    {fam:20s} {fila['id']}  {fila['parametros']}  CV {fila['cv_media']:.2f}%")

# Modelo de la app (parámetros por defecto), para que se pueda rastrear
app_modelo = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
p_app = cross_val_score(app_modelo, X_tr, y_tr, groups=g_tr, cv=cv, scoring="accuracy")
print(f"\n  Modelo actual de la app (Random Forest, 300 árboles, valores por defecto): "
      f"CV {p_app.mean() * 100:.2f}% (±{p_app.std() * 100:.2f})")
resumen["modelo_app_cv"] = round(p_app.mean() * 100, 2)
resumen["mejores"] = {f: {"id": r["id"], "parametros": r["parametros"], "cv": float(r["cv_media"])}
                      for f, (r, _) in mejores.items()}

# ====================================================== FASE 4: EVALUACIÓN EN PRUEBA
print("\n=== FASE 4: evaluación en el conjunto de prueba (usado una sola vez) ===")
filas = []
ajustados = {}
base = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
candidatos = [("Baseline (always the majority class)", base, "-")]
for fam, (fila, mod) in mejores.items():
    candidatos.append((fam, mod, fila["parametros"]))
candidatos.append(("Random Forest (app model)", app_modelo, "trees=300, defaults"))
for nombre, mod, par in candidatos:
    mod.fit(X_tr, y_tr)
    ajustados[nombre] = mod
    pred = mod.predict(X_te)
    pr, rc, f1, _ = precision_recall_fscore_support(y_te, pred, labels=[1], zero_division=0)
    auc = roc_auc_score(y_te, mod.predict_proba(X_te)[:, 1]) if hasattr(mod, "predict_proba") and nombre.startswith("Baseline") is False else np.nan
    cvm = (tabla_exp.loc[tabla_exp["parametros"] == par, "cv_media"].max() if par != "-" else np.nan)
    if nombre.startswith("Random Forest (app"):
        cvm = p_app.mean() * 100
    filas.append({"modelo": nombre, "parametros": par,
                  "exactitud_train": round(accuracy_score(y_tr, mod.predict(X_tr)) * 100, 2),
                  "cv_media": round(float(cvm), 2) if not np.isnan(cvm) else np.nan,
                  "exactitud_test": round(accuracy_score(y_te, pred) * 100, 2),
                  "precision_enferma": round(pr[0] * 100, 1), "recall_enferma": round(rc[0] * 100, 1),
                  "f1_enferma": round(f1[0] * 100, 1), "auc": round(auc, 3) if not np.isnan(auc) else np.nan})
comp = pd.DataFrame(filas)
comp.to_csv(OUT / "comparacion_modelos.csv", index=False)
print(comp.to_string(index=False))
resumen["comparacion"] = comp.replace({np.nan: None}).to_dict("records")

# Modelo seleccionado. La validación cruzada la gana Gradient Boosting por 0.4 puntos, una diferencia menor
# que la variación entre partes (±2 puntos); en la prueba empatan. Se mantiene el Random Forest de la app
# y esa decisión se explica en el informe (no es un resultado, es un criterio razonado).
ganador_cv = max(mejores, key=lambda f: mejores[f][0]["cv_media"])
print(f"\n  Mejor validación cruzada: {ganador_cv} ({mejores[ganador_cv][0]['parametros']})")
fam_sel = "Random Forest (app model)"
print(f"  Modelo seleccionado para el informe y la app: {fam_sel}")
resumen["ganador_validacion_cruzada"] = ganador_cv
resumen["familia_seleccionada"] = fam_sel
sel = ajustados[fam_sel]
pred_sel = sel.predict(X_te)
rep = precision_recall_fscore_support(y_te, pred_sel, labels=[0, 1], zero_division=0)
por_clase = pd.DataFrame({"clase": ["Healthy", "Sick"], "precision": (rep[0] * 100).round(1),
                          "recall": (rep[1] * 100).round(1), "f1": (rep[2] * 100).round(1),
                          "fotos": rep[3]})
print("\n  Precision, recall y F1 del modelo seleccionado:")
print(por_clase.to_string(index=False))
por_clase.to_csv(OUT / "precision_recall.csv", index=False)
resumen["precision_recall"] = por_clase.to_dict("records")

cm = confusion_matrix(y_te, pred_sel)
fig, ax = plt.subplots(figsize=(4.6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False, ax=ax,
            xticklabels=["Healthy", "Sick"], yticklabels=["Healthy", "Sick"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Real")
ax.set_title("Test set: Random Forest (app model)")
guardar(fig, "fig_matriz_confusion.png")
resumen["matriz_confusion"] = cm.tolist()

# Acierto por clase y sesgo
prueba = data.iloc[i_te].copy()
prueba["acierto"] = (pred_sel == y_te.values)
cl = (prueba.groupby("clase")["acierto"].agg(["mean", "count"]).reindex(orden))
cl["mean"] = (cl["mean"] * 100).round(1)
cl.columns = ["acierto_pct", "fotos"]
cl.to_csv(OUT / "acierto_por_clase.csv")
print("\n  Acierto por clase en la prueba:")
print(cl.to_string())
resumen["acierto_por_clase"] = cl.reset_index().to_dict("records")

# Sobreajuste: curva de aprendizaje con el modelo seleccionado
tam, tr_s, va_s = learning_curve(app_modelo, X_tr, y_tr, groups=g_tr, cv=cv,
                                 train_sizes=np.linspace(0.2, 1.0, 5), scoring="accuracy")
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(tam, tr_s.mean(axis=1) * 100, "o-", color="#A5531A", label="Training")
ax.plot(tam, va_s.mean(axis=1) * 100, "o-", color="#2F6B47", label="Validation (grouped by plant)")
ax.set_xlabel("Training photos")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Learning curve")
ax.legend()
guardar(fig, "fig_curva_aprendizaje.png")
resumen["curva_aprendizaje"] = {"tamanos": tam.tolist(), "train": (tr_s.mean(axis=1) * 100).round(1).tolist(),
                                "validacion": (va_s.mean(axis=1) * 100).round(1).tolist()}

fig, ax = plt.subplots(figsize=(8, 4))
c2 = comp[~comp["modelo"].str.startswith("Baseline")]
x = np.arange(len(c2))
ax.bar(x - 0.2, c2["cv_media"], 0.4, label="Cross-validation (train)", color="#2F6B47")
ax.bar(x + 0.2, c2["exactitud_test"], 0.4, label="Test", color="#E39A55")
ax.set_xticks(x)
ax.set_xticklabels([m.replace(" (", "\n(") for m in c2["modelo"]], fontsize=8)
ax.axhline(80, color="#A5531A", ls="--", lw=1)
ax.text(len(c2) - 0.5, 80.5, "goal: 80%", ha="right", color="#A5531A", fontsize=9)
ax.set_ylim(50, 100)
ax.set_ylabel("Accuracy (%)")
ax.legend(loc="lower right")
guardar(fig, "fig_comparacion.png")

# Referencia con contorno del experto (misma división)
Xe, ye = exp[car.NOMBRES], exp["target"]
rf_e = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).fit(Xe.iloc[i_tr], ye.iloc[i_tr])
acc_e = accuracy_score(ye.iloc[i_te], rf_e.predict(Xe.iloc[i_te])) * 100
print(f"\n  Referencia con el contorno del experto (misma división): {acc_e:.2f}%")
resumen["test_con_contorno_experto"] = round(acc_e, 2)

# Otras tareas del proyecto: enfermedad (KNN) y nivel de roya (regresión lineal)
knn6 = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)).fit(X_tr, data["clase"].iloc[i_tr])
acc6 = accuracy_score(data["clase"].iloc[i_te], knn6.predict(X_te)) * 100
roya_tr = data.iloc[i_tr][data["clase"].iloc[i_tr].str.startswith("rust")]
roya_te = data.iloc[i_te][data["clase"].iloc[i_te].str.startswith("rust")]
nivel = lambda d: d["clase"].str[-1].astype(int)
reg = make_pipeline(StandardScaler(), LinearRegression()).fit(roya_tr[car.NOMBRES], nivel(roya_tr))
pr_n = reg.predict(roya_te[car.NOMBRES])
r2 = r2_score(nivel(roya_te), pr_n)
mae = mean_absolute_error(nivel(roya_te), pr_n)
print(f"  KNN (6 clases, k=7), exactitud en prueba: {acc6:.1f}%")
print(f"  Regresión lineal (nivel de roya 1-4), R² en prueba: {r2:.2f} | error medio: {mae:.2f} niveles")
resumen["otras_tareas"] = {"knn_6_clases_test": round(acc6, 1), "regresion_r2": round(r2, 2),
                           "regresion_mae": round(mae, 2), "fotos_roya_test": int(len(roya_te))}

# ====================================================== FASE 6: PRUEBA DE MONITOREO (datos simulados)
print("\n=== FASE 6: prueba de monitoreo (la parte B es SIMULADA) ===")


def psi(ref, nuevo, bins=10):
    cortes = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf
    a = np.histogram(ref, cortes)[0] / len(ref)
    b = np.histogram(nuevo, cortes)[0] / len(nuevo)
    a, b = np.clip(a, 1e-4, None), np.clip(b, 1e-4, None)
    return float(np.sum((b - a) * np.log(b / a)))


ref = {"media": X_tr.mean().round(5).to_dict(), "desv": X_tr.std().round(5).to_dict(),
       "fotos_entrenamiento": len(X_tr)}
(MODELOS / "referencia_monitoreo.json").write_text(json.dumps(ref, indent=1))

if hasattr(sel, "feature_importances_"):
    importantes = list(pd.Series(sel.feature_importances_, index=car.NOMBRES).nlargest(5).index)
else:
    importantes = car.NOMBRES[:5]
X_sim = X_te.copy()
for c in importantes:
    X_sim[c] = X_sim[c] + 1.5 * X_tr[c].std()        # SIMULADO: cambio de iluminación o de cámara

registro = []
resumen_mon = []
for esc_nombre, Xb, simulado in [("A: held-out test photos (real)", X_te, False),
                                 ("B: shifted batch (SIMULATED)", X_sim, True)]:
    probs = sel.predict_proba(Xb)[:, 1]
    pred = (probs >= 0.5).astype(int)
    psis = {c: psi(X_tr[c].values, Xb[c].values) for c in car.NOMBRES}
    ks_alarmas = sum(ks_2samp(X_tr[c], Xb[c]).pvalue < 0.01 for c in car.NOMBRES)
    conf = np.maximum(probs, 1 - probs)
    resumen_mon.append({"escenario": esc_nombre, "simulado": simulado, "fotos": len(Xb),
                        "psi_max": round(max(psis.values()), 3),
                        "caracteristicas_psi_mayor_0_2": int(sum(v > 0.2 for v in psis.values())),
                        "caracteristicas_ks_p_menor_0_01": int(ks_alarmas),
                        "confianza_media": round(float(conf.mean()), 3),
                        "exactitud": round(accuracy_score(y_te, pred) * 100, 1),
                        "psi": psis})
    for a, p, pr_ in zip(data["archivo"].iloc[i_te], probs, pred):
        registro.append({"escenario": esc_nombre, "simulado": simulado, "archivo": a,
                         "prob_enferma": round(float(p), 4), "prediccion": int(pr_),
                         "confianza": round(float(max(p, 1 - p)), 4)})
pd.DataFrame(registro).to_csv(OUT / "monitoreo_simulado.csv", index=False)
tabla_mon = pd.DataFrame([{k: v for k, v in m.items() if k != "psi"} for m in resumen_mon])
tabla_mon.to_csv(OUT / "monitoreo_resumen.csv", index=False)
print(tabla_mon.to_string(index=False))
print(f"  Características con más cambio en B (las que se movieron a propósito): {importantes}")
resumen["monitoreo"] = {"filas": tabla_mon.to_dict("records"), "caracteristicas_movidas_simulacion": importantes}

fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(len(car.NOMBRES))
ax.bar(x - 0.2, [resumen_mon[0]["psi"][c] for c in car.NOMBRES], 0.4, label="A: test photos (real)", color="#2F6B47")
ax.bar(x + 0.2, [resumen_mon[1]["psi"][c] for c in car.NOMBRES], 0.4, label="B: shifted batch (SIMULATED)", color="#E39A55")
ax.axhline(0.2, color="#A5531A", ls="--", lw=1)
ax.text(len(x) - 0.5, 0.21, "alarm: PSI > 0.2", ha="right", color="#A5531A", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(car.NOMBRES, rotation=90, fontsize=7)
ax.set_ylabel("PSI vs. training data")
ax.legend()
guardar(fig, "fig_deriva.png")

# 4. Guardar el resumen
(OUT / "resumen.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False, default=str))
print("\nListo. Todo quedó en la carpeta evidencia/ (tablas .csv, gráficas .png y resumen.json).")
print("El archivo models/referencia_monitoreo.json es nuevo; el modelo de la app NO se modificó.")