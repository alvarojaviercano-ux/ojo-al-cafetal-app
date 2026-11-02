# Paso 13 - Q-Learning: ¿qué hacer según el estado de la hoja?
# Mismo esquema del ejemplo del profesor (reinforcement.py): estados, acciones,
# step(), encode(), SGDRegressor con partial_fit y política epsilon-greedy.
#
# IMPORTANTE: es una SIMULACIÓN. Las probabilidades y recompensas son supuestos
# (no medidos). Las reglas de decisión siguen la guía de Cenicafé sobre roya:
#  - variedades resistentes (Castillo, Colombia, Cenicafé 1, Tabi): no se recomienda control químico
#  - variedades susceptibles (Caturra, Típica, Borbón): controlar antes de que la roya avance,
#    con fungicida sistémico (2 o 3 aplicaciones por ciclo)

# 1. Importar librerías necesarias
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor

RAIZ = Path(__file__).resolve().parent
MODELOS = RAIZ / "models"
MODELOS.mkdir(exist_ok=True)

# 2. Estados: (nivel de roya 0-4, variedad). Nivel 0 = hoja sana.
NIVELES = ["sana", "roya nivel 1", "roya nivel 2", "roya nivel 3", "roya nivel 4"]
VARIEDADES = ["resistente", "susceptible"]
N_NIVELES = len(NIVELES)
N_ESTADOS = N_NIVELES * len(VARIEDADES)


def estado(nivel, variedad):
    """Convierte (nivel, variedad) en un número de estado."""
    return variedad * N_NIVELES + nivel


def desarmar(s):
    return s % N_NIVELES, s // N_NIVELES      # (nivel, variedad)


# 3. Acciones
ACTIONS = [0, 1, 2]
ACTION_NAMES = ["Solo monitorear", "Mejorar el manejo del cultivo", "Aplicar fungicida"]
N_ACCIONES = len(ACTIONS)

# 4. Supuestos de la simulación (cada paso = un mes)
# Probabilidad de que la roya suba un nivel / baje un nivel, según la acción.
#                     acción 0 (monitorear)  1 (manejo)    2 (fungicida)
P_SUBE = {"resistente":  [0.08, 0.04, 0.02],
          "susceptible": [0.40, 0.20, 0.05]}
P_BAJA = {"resistente":  [0.05, 0.25, 0.40],
          "susceptible": [0.03, 0.15, 0.55]}
INGRESO_POR_NIVEL = [10, 8, 5, 1, -6]     # lo que deja la producción del mes según el nivel
COSTO_ACCION = [0.0, 1.5, 3.0]            # costo de cada acción
PENALIZACION_FUNGICIDA_RESISTENTE = 6.0   # Cenicafé no lo recomienda en variedades resistentes


def step(s, accion, rng):
    """Aplica una acción y devuelve (estado nuevo, recompensa)."""
    nivel, variedad = desarmar(s)
    nombre = VARIEDADES[variedad]
    azar = rng.random()
    if nivel < N_NIVELES - 1 and azar < P_SUBE[nombre][accion]:
        nivel += 1
    elif nivel > 0 and azar < P_SUBE[nombre][accion] + P_BAJA[nombre][accion]:
        nivel -= 1
    recompensa = INGRESO_POR_NIVEL[nivel] - COSTO_ACCION[accion]
    if accion == 2 and nombre == "resistente":
        recompensa -= PENALIZACION_FUNGICIDA_RESISTENTE
    return estado(nivel, variedad), recompensa


# 5. Codificar (estado, acción) como un vector de ceros y un uno
def encode(s, accion):
    vector = np.zeros(N_ESTADOS * N_ACCIONES)
    vector[s * N_ACCIONES + accion] = 1.0
    return vector.reshape(1, -1)


# 6. Modelo: SGDRegressor que aprende el valor Q de cada (estado, acción)
q_model = SGDRegressor(loss="squared_error", penalty=None, fit_intercept=False,
                       learning_rate="constant", eta0=0.1, random_state=42)
q_model.partial_fit(encode(0, 0), [0.0])      # inicializar con ceros


def predict_q_values(s):
    # El vector de entrada tiene un solo 1, así que el valor Q de (estado, acción)
    # es simplemente su peso en el modelo. Es lo mismo que q_model.predict(...)
    # pero mucho más rápido.
    pesos = q_model.coef_[s * N_ACCIONES:(s + 1) * N_ACCIONES]
    return [float(p) for p in pesos]


# 7. Entrenamiento
def train(episodes=8000, pasos_por_episodio=12, gamma=0.95):
    rng = random.Random(42)
    epsilon = 1.0
    for episodio in range(episodes):
        s = rng.randrange(N_ESTADOS)           # empieza en un estado al azar
        for paso in range(pasos_por_episodio):    # 12 meses
            # La primera acción de cada episodio es al azar, para probar todas las
            # combinaciones (incluso en estados raros, como roya nivel 4 en variedad resistente)
            if paso == 0 or rng.random() < epsilon:
                accion = rng.choice(ACTIONS)
            else:
                accion = int(np.argmax(predict_q_values(s)))
            s_nuevo, recompensa = step(s, accion, rng)
            objetivo = recompensa + gamma * max(predict_q_values(s_nuevo))
            q_model.partial_fit(encode(s, accion), [objetivo])
            s = s_nuevo
        epsilon = max(0.05, epsilon * 0.9993)


train()

# 8. Política aprendida: la mejor acción en cada estado
filas = []
for variedad, nombre_var in enumerate(VARIEDADES):
    for nivel, nombre_nivel in enumerate(NIVELES):
        s = estado(nivel, variedad)
        q = predict_q_values(s)
        filas.append({"variedad": nombre_var, "estado": nombre_nivel,
                      "mejor_accion": ACTION_NAMES[int(np.argmax(q))],
                      "Q_monitorear": round(q[0], 1), "Q_manejo": round(q[1], 1),
                      "Q_fungicida": round(q[2], 1)})
politica = pd.DataFrame(filas)
print(politica.to_string(index=False))

# 9. Guardar para la app
joblib.dump({"q_model": q_model, "niveles": NIVELES, "variedades": VARIEDADES,
             "acciones": ACTION_NAMES}, MODELOS / "qlearning.joblib")
print("\nModelo guardado en models/qlearning.joblib")