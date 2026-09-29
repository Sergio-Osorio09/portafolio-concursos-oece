"""Sistema difuso 1: compatibilidad empresa-concurso (0-100), UNA evaluación por concurso.

Algunas variables no vienen en los datos abiertos y se estiman con supuestos explícitos
(personal requerido, duración, experiencia requerida). Se muestran en la interfaz.
"""
import numpy as np
import pandas as pd

from .difuso import SistemaMamdani

VARIABLES = {
    "ratio_monto": {"pequeno": (0, 0, 0.25), "adecuado": (0.1, 0.45, 0.8), "exigente": (0.6, 1, 1)},
    "experiencia": {"insuficiente": (0, 0, 0.8), "aceptable": (0.5, 1, 1.5), "fuerte": (1.2, 2, 2)},
    "plazo":       {"ajustado": (0, 0, 7), "razonable": (4, 10, 18), "holgado": (14, 30, 30)},
    "tecnica":     {"baja": (0, 0, 0.4), "media": (0.2, 0.5, 0.8), "alta": (0.6, 1, 1)},
    "personal":    {"limitada": (0, 0, 1), "suficiente": (0.7, 1.2, 2), "amplia": (1.6, 3, 3)},
    "ubicacion":   {"baja": (0, 0, 0.5), "media": (0.3, 0.55, 0.8), "alta": (0.6, 1, 1)},
}
UNIVERSOS = {"ratio_monto": (0, 1), "experiencia": (0, 2), "plazo": (0, 30),
             "tecnica": (0, 1), "personal": (0, 3), "ubicacion": (0, 1)}
DESCRIPCION = {
    "ratio_monto": "monto del concurso / capacidad financiera",
    "experiencia": "años de la empresa / años requeridos estimados (1 + monto/300 000)",
    "plazo": "días hasta el cierre estimado de postulación",
    "tecnica": "afinidad textual con el rubro (palabras clave encontradas / 3)",
    "personal": "personal disponible / personal estimado del concurso",
    "ubicacion": "1 mismo departamento · 0.7 en cobertura · 0.5 desconocido · 0.3 otro",
}
SALIDA = {"baja": (0, 0, 40), "media": (25, 50, 70), "alta": (55, 75, 90), "muy_alta": (80, 100, 100)}

REGLAS = [
    ({"tecnica": "alta", "experiencia": "fuerte", "ratio_monto": "adecuado"}, "muy_alta"),
    ({"tecnica": "alta", "ubicacion": "alta", "plazo": "holgado"}, "muy_alta"),
    ({"tecnica": "alta", "experiencia": "aceptable", "personal": "suficiente"}, "alta"),
    ({"tecnica": "alta", "plazo": "razonable"}, "alta"),
    ({"tecnica": "alta", "ratio_monto": "pequeno"}, "alta"),
    ({"tecnica": "media", "ubicacion": "alta", "personal": "amplia"}, "alta"),
    ({"tecnica": "media", "experiencia": "fuerte", "personal": "amplia"}, "alta"),
    ({"tecnica": "media", "experiencia": "aceptable"}, "media"),
    ({"ratio_monto": "exigente", "experiencia": "fuerte", "personal": "amplia"}, "alta"),
    ({"plazo": "ajustado", "tecnica": "alta", "personal": "amplia"}, "media"),
    ({"ubicacion": "baja", "tecnica": "alta"}, "media"),
    ({"tecnica": "baja"}, "baja"),
    ({"experiencia": "insuficiente"}, "baja"),
    ({"ratio_monto": "exigente", "personal": "limitada"}, "baja"),
    ({"plazo": "ajustado", "personal": "limitada"}, "baja"),
    ({"ubicacion": "baja", "ratio_monto": "exigente"}, "baja"),
]

SISTEMA = SistemaMamdani(VARIABLES, SALIDA, REGLAS, puntos=501)
ENTRADAS = list(VARIABLES)


def estimar_personal(monto, cat, ficha):
    # obras requieren más gente; escala con el personal típico del rubro de la empresa
    base = {"goods": 1, "services": 2, "works": 4}.get(cat, 2)
    paso = 250_000 if cat != "works" else 400_000
    return int(np.clip(base + monto // paso, 1, max(8, ficha["personal_disponible"])))


def estimar_duracion(monto, cat, dur):
    if pd.notna(dur) and dur > 0:
        return int(dur)
    lo, hi = {"goods": (30, 90), "services": (60, 180), "works": (90, 240)}.get(cat, (60, 180))
    return int(np.clip(lo + monto / 10_000, lo, hi))


def puntaje_ubicacion(dep, ficha):
    if dep == ficha["departamento_base"]:
        return 1.0
    if dep in ficha["departamentos_cobertura"]:
        return 0.7
    if dep == "DESCONOCIDO":
        return 0.5
    return 0.3


def lectura(p):
    if p >= 85: return "Muy alta"
    if p >= 75: return "Alta"
    if p >= 60: return "Buena"
    if p >= 40: return "Media"
    return "Baja"


def evaluar(factibles, ficha, fecha_ref):
    fac = factibles.copy()
    fac["personal_req"] = [estimar_personal(m, c, ficha) for m, c in zip(fac["monto_pen"], fac["categoria"])]
    fac["duracion_est"] = [estimar_duracion(m, c, d) for m, c, d in zip(fac["monto_pen"], fac["categoria"], fac["duracion_dias"])]
    fac["exp_req"] = 1 + fac["monto_pen"] / 300_000

    fac["ratio_monto"] = (fac["monto_pen"] / ficha["capacidad_financiera"]).clip(0, 1)
    fac["experiencia"] = (ficha["experiencia_anios"] / fac["exp_req"]).clip(0, 2)
    fac["plazo"] = (fac["cierre_est"] - fecha_ref).dt.days.clip(0, 30).fillna(0)
    fac["tecnica"] = fac["afinidad"]
    fac["personal"] = (ficha["personal_disponible"] / fac["personal_req"]).clip(0, 3)
    fac["ubicacion"] = fac["departamento"].map(lambda d: puntaje_ubicacion(d, ficha))

    scores, dominantes = [], []
    for fila in fac[ENTRADAS].to_dict("records"):
        r = SISTEMA.evaluar(fila, detalle=True)
        scores.append(round(r["valor"], 1))
        dominantes.append(r["activas"][0] if r["activas"] else None)
    fac["score"] = scores
    fac["regla_dominante"] = dominantes
    fac["lectura"] = fac["score"].map(lectura)
    return fac.sort_values("score", ascending=False)
