"""Filtro determinístico (restricciones duras).

Descarta lo que NO admite gradualidad: si el concurso es de otro rubro o supera la capacidad
financiera, no tiene sentido evaluarlo con lógica difusa. Así el motor difuso solo trabaja
sobre concursos factibles y la explicación de cada descarte es exacta.
"""
import re

import numpy as np
import pandas as pd

from .oece import norm

TIPO_CAMBIO_USD = 3.75
ESTADOS_INVALIDOS = {"NULO", "CANCELADO", "DESIERTO", "SUSPENDIDO", "ADJUDICADO", "CONSENTIDO", "CONTRATADO"}


def patrones(ficha):
    return [re.compile(r"\b" + re.escape(norm(k)) + r"\b") for k in ficha["rubro_keywords"] if norm(k)]


def afinidad(texto, pats):
    t = norm(texto)
    return min(1.0, sum(1 for p in pats if p.search(t)) / 3)   # 3 o más coincidencias = afinidad máxima


def estimar_montos(df, montos_hist):
    """Monto reservado (0 o vacío) -> mediana histórica de adjudicados del mismo método/categoría."""
    reservado = ~(df["monto_pen"] > 0)
    est = [montos_hist["metodo"].get((c, m), montos_hist["categoria"].get(c, np.nan))
           for c, m in zip(df["categoria"], df["metodo"])]
    df["monto_estimado"] = reservado & pd.Series(est, index=df.index).notna()
    df.loc[reservado, "monto_pen"] = pd.Series(est, index=df.index)[reservado]
    return df


def aplicar(vigentes, ficha, montos_hist=None):
    """Devuelve (factibles, resumen del embudo). Si se pasa `montos_hist`, los montos reservados se
    estiman en lugar de descartarse."""
    df = vigentes.copy()
    df["monto_pen"] = np.where(df["moneda"] == "USD", df["monto"] * TIPO_CAMBIO_USD,
                               np.where(df["moneda"].isna() | (df["moneda"] == "PEN"), df["monto"], np.nan))
    df["monto_estimado"] = False
    if montos_hist is not None:
        df = estimar_montos(df, montos_hist)
    pats = patrones(ficha)
    df["afinidad"] = (df["titulo"].fillna("") + " " + df["descripcion"].fillna("") + " "
                      + df["texto_items"].fillna("")).map(lambda t: afinidad(t, pats))
    metodo_n = df["metodo"].map(norm)

    reglas = {
        "Estado inválido (nulo/desierto/adjudicado...)": ~df["estado_detalle"].fillna("").str.upper().isin(ESTADOS_INVALIDOS),
        "Categoría no atendida": df["categoria"].isin(ficha["categorias"]),
        "Contratación directa": ~metodo_n.str.contains("contratacion directa", na=False),
        "Monto no disponible": df["monto_pen"].notna() & (df["monto_pen"] > 0),
        "Monto supera capacidad financiera": df["monto_pen"] <= ficha["capacidad_financiera"],
        "Monto menor al de interés": df["monto_pen"] >= ficha["monto_minimo_interes"],
        "Rubro ajeno (sin palabras clave)": df["afinidad"] > 0,
    }
    if ficha["cobertura"] != "nacional":
        reglas["Fuera de cobertura geográfica"] = df["departamento"].isin(list(ficha["departamentos_cobertura"]) + ["DESCONOCIDO"])

    mascara = pd.Series(True, index=df.index)
    embudo = [("Concursos vigentes", len(df))]
    for nombre, regla in reglas.items():
        mascara &= regla.fillna(False)
        embudo.append((nombre, int(mascara.sum())))
    return df[mascara].copy(), pd.DataFrame(embudo, columns=["Etapa", "Quedan"])
