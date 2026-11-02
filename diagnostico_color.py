# Diagnóstico: ¿la regla de color ve las manchas en la mayoría de las fotos?

from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent
data = pd.read_csv(RAIZ / "data.csv")

orden = [c for c in ["healthy", "rust_level_1", "rust_level_2", "rust_level_3",
                     "rust_level_4", "red_spider_mite"] if c in set(data["clase"])]

for columna, nombre in [("fraccion_amarillo_naranja", "Píxeles amarillos/naranjas/café (%)"),
                        ("fraccion_manchas", "Manchas conectadas (%)")]:
    print(f"\n{nombre}, por clase:")
    tabla = (data.groupby("clase")[columna]
             .quantile([0.25, 0.5, 0.75]).unstack() * 100).loc[orden]
    tabla.columns = ["25%", "mediana", "75%"]
    tabla["fotos con más de 1%"] = (data.groupby("clase")[columna]
                                    .apply(lambda s: (s > 0.01).mean() * 100)).loc[orden]
    print(tabla.round(1).to_string())

print("\nTono (H) promedio de la hoja, por clase (escala 0-255; verde ≈ 60-90):")
print(data.groupby("clase")["media_H"].mean().loc[orden].round(1).to_string())