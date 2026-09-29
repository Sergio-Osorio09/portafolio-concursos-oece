"""Algoritmo genético del portafolio, con el riesgo difuso DENTRO del fitness.

- Gen: x_i ∈ {0,1} -> postular o no al concurso candidato C_i.
- Cromosoma: cadena binaria de longitud N = un portafolio completo.
- Fitness: Σ x_i·s_i combinado con Riesgo_difuso(x) como resta o como descuento (ver Evaluador),
  opcionalmente con las penalidades clásicas de recursos, fechas y ventana.
- Restricciones duras (vía reparación): 1 ≤ Σx ≤ 2·capacidad y costo total ≤ presupuesto.
- Operadores: torneo, cruce uniforme, mutación bit a bit, elitismo, reparación, parada por paciencia.
"""
import itertools
import math

import numpy as np
import pandas as pd

from . import riesgo as R

COSTO_FIJO = 2_500          # S/ por preparar una propuesta
PCT_COSTO = 0.06            # % del monto (garantías / capital de trabajo)
DIAS_HASTA_EJECUCION = 20   # días entre cierre de postulación e inicio de ejecución
DIAS_CHOQUE = 7             # dos cierres a menos de esta distancia compiten por el mismo equipo de propuestas

AG_DEFECTO = {"poblacion": 80, "generaciones": 150, "p_cruce": 0.9, "torneo": 3, "elite": 2, "paciencia": 40}


def choque_cierres(cierre_dias):
    """conflicto(i,j) = 1 si cierran el mismo día, baja linealmente a 0 a los DIAS_CHOQUE días.

    Nota de diseño: el notebook medía el solape de EJECUCIÓN, pero con datos reales todos los
    concursos vigentes se ejecutan en el mismo periodo (conflicto ≈ 1 para cualquier par). Esa
    simultaneidad ya la mide `cantidad / capacidad operativa` (proyectos simultáneos que la empresa
    soporta), así que aquí `fechas` mide el choque de PREPARACIÓN: tener que armar varias propuestas
    para la misma semana."""
    d = np.abs(cierre_dias[:, None] - cierre_dias[None, :])
    return np.clip(1 - d / DIAS_CHOQUE, 0, 1)


def construir_contexto(cand, ficha, fecha_ref):
    """Precalcula todo lo que el fitness necesita (vectores y matriz de conflicto)."""
    n = len(cand)
    dia = 86_400e9
    prep_ini = np.full(n, fecha_ref.value, dtype=np.int64)
    prep_fin = np.maximum(cand["cierre_est"].values.astype("datetime64[ns]").astype(np.int64), prep_ini + np.int64(dia))
    ejec_ini = prep_fin + np.int64(DIAS_HASTA_EJECUCION * dia)
    ejec_fin = ejec_ini + (cand["duracion_est"].values * dia).astype(np.int64)

    conflicto = choque_cierres(prep_fin / dia)
    np.fill_diagonal(conflicto, 0)

    v_ini, v_fin = [pd.Timestamp(d).value for d in ficha["ventana_disponible"]]
    dentro = np.maximum(0, np.minimum(ejec_fin, v_fin) - np.maximum(ejec_ini, v_ini))

    return {
        "n": n,
        "s": cand["score"].values.astype(float),
        "monto": cand["monto_pen"].values.astype(float),
        "costo": (COSTO_FIJO + PCT_COSTO * cand["monto_pen"]).values.astype(float),
        "pers": cand["personal_req"].values.astype(float),
        "conflicto": conflicto,
        "conf_sup": np.triu(conflicto, 1),
        "fuera": 1 - dentro / np.maximum(1, ejec_fin - ejec_ini),
        "cierre": pd.to_datetime(prep_fin),
        "ejec_ini": pd.to_datetime(ejec_ini), "ejec_fin": pd.to_datetime(ejec_fin),
        "capacidad_operativa": ficha["capacidad_operativa"],
        "capacidad_financiera": ficha["capacidad_financiera"],
        "personal_disponible": ficha["personal_disponible"],
        "presupuesto": ficha["presupuesto_postulaciones"],
        "k_max": min(n, 2 * ficha["capacidad_operativa"]),
    }


class Evaluador:
    """Fitness con memoria (el mismo portafolio no se reevalúa).

    forma="resta"     : F = Σs − λ·R                     (versión del notebook)
    forma="descuento" : F = Σs · (1 − R/100)^λ           (valor esperado: el riesgo descuenta el beneficio)
    penalidades=True agrega además − λ_rec·P_recursos − λ_fech·P_fechas − λ_vent·P_ventana.

    Por qué existe "descuento": con la resta, Σs crece sin límite con cada licitación pero R se
    satura en 100, así que λ produce un salto todo-o-nada (6 licitaciones -> 1). Con el descuento
    cada licitación extra solo conviene si su aporte supera lo que el riesgo le quita a TODO el
    portafolio, y el nº de licitaciones baja de forma gradual al subir λ.
    """

    def __init__(self, ctx, lambdas, forma="descuento", penalidades=False):
        self.ctx, self.lam, self.forma, self.penalidades = ctx, lambdas, forma, penalidades
        self.cache, self.evaluaciones = {}, 0

    def descomposicion(self, x):
        """Aporte de cada término al fitness (positivo = suma, negativo = resta)."""
        c, lam = self.ctx, self.lam
        s = float(c["s"] @ x)
        r = R.riesgo(x, c)
        d = {"Σ compatibilidad": s}
        if self.forma == "resta":
            d["− λ·riesgo"] = -lam["riesgo"] * r
        else:
            d["− pérdida por riesgo"] = -s * (1 - (1 - r / 100) ** lam["riesgo"])
        if self.penalidades:
            d["− λ·P_recursos"] = -lam.get("recursos", 0) * max(0.0, c["pers"] @ x - c["personal_disponible"]) / c["personal_disponible"] * 100
            d["− λ·P_fechas"] = -lam.get("fechas", 0) * float(x @ c["conf_sup"] @ x) * 100
            d["− λ·P_ventana"] = -lam.get("ventana", 0) * float(c["fuera"] @ x) * 100
        return d

    def __call__(self, x):
        key = x.tobytes()
        if key not in self.cache:
            self.evaluaciones += 1
            self.cache[key] = sum(self.descomposicion(x).values())
        return self.cache[key]


def reparar(x, ctx):
    x = x.copy()
    eficiencia = ctx["s"] / ctx["costo"]
    while x.sum() > ctx["k_max"] or ctx["costo"] @ x > ctx["presupuesto"]:
        act = np.flatnonzero(x)
        x[act[np.argmin(eficiencia[act])]] = 0
    if x.sum() == 0:
        for i in np.argsort(-ctx["s"]):
            if ctx["costo"][i] <= ctx["presupuesto"]:
                x[i] = 1
                break
    return x


def algoritmo_genetico(ctx, fit, params=None, seed=42):
    p = {**AG_DEFECTO, **(params or {})}
    n = ctx["n"]
    rng = np.random.default_rng(seed)
    p_mut = p.get("p_mut") or 1 / n

    pob = (rng.random((p["poblacion"], n)) < ctx["capacidad_operativa"] / n).astype(np.int8)
    pob = np.array([reparar(x, ctx) for x in pob])
    fits = np.array([fit(x) for x in pob])
    hist = {"mejor": [], "media": [], "k_mejor": []}
    mejor_global, sin_mejora, gen = -np.inf, 0, 0

    def torneo():
        idx = rng.choice(len(pob), p["torneo"], replace=False)
        return pob[idx[np.argmax(fits[idx])]]

    for gen in range(1, p["generaciones"] + 1):
        nueva = [pob[i].copy() for i in np.argsort(-fits)[:p["elite"]]]          # elitismo
        while len(nueva) < p["poblacion"]:
            p1, p2 = torneo(), torneo()                                            # selección
            if rng.random() < p["p_cruce"]:                                        # cruce uniforme
                m = rng.random(n) < 0.5
                h1, h2 = np.where(m, p1, p2), np.where(m, p2, p1)
            else:
                h1, h2 = p1.copy(), p2.copy()
            for h in (h1, h2):
                flip = rng.random(n) < p_mut                                       # mutación
                h[flip] = 1 - h[flip]
                nueva.append(reparar(h.astype(np.int8), ctx))                      # reparación
        pob = np.array(nueva[:p["poblacion"]])
        fits = np.array([fit(x) for x in pob])

        b = int(np.argmax(fits))
        hist["mejor"].append(float(fits[b])); hist["media"].append(float(fits.mean()))
        hist["k_mejor"].append(int(pob[b].sum()))
        if fits[b] > mejor_global + 1e-9:
            mejor_global, sin_mejora = fits[b], 0
        else:
            sin_mejora += 1
            if sin_mejora >= p["paciencia"]:
                break

    unicos = {}
    for x, fx in zip(pob, fits):
        unicos.setdefault(x.tobytes(), (x, fx))
    ranking = sorted(unicos.values(), key=lambda t: -t[1])
    return {"ranking": ranking, "hist": hist, "generaciones": gen, "evaluaciones": fit.evaluaciones}


def n_combinaciones(ctx):
    return sum(math.comb(ctx["n"], k) for k in range(1, ctx["k_max"] + 1))


def exhaustivo(ctx, fit):
    mejor, xb, evaluados = -np.inf, None, 0
    for k in range(1, ctx["k_max"] + 1):
        for comb in itertools.combinations(range(ctx["n"]), k):
            x = np.zeros(ctx["n"], dtype=np.int8); x[list(comb)] = 1
            if ctx["costo"] @ x > ctx["presupuesto"]:
                continue
            evaluados += 1
            f = fit(x)
            if f > mejor:
                mejor, xb = f, x
    return xb, mejor, evaluados


def barrido_lambda(ctx, lambdas_base, valores, forma="descuento", penalidades=False, params=None):
    """Para cada λ_riesgo corre el AG y registra cuántas licitaciones elige y con qué riesgo."""
    filas = []
    for lam in valores:
        fit = Evaluador(ctx, {**lambdas_base, "riesgo": lam}, forma, penalidades)
        res = algoritmo_genetico(ctx, fit, params)
        x = res["ranking"][0][0]
        filas.append({"λ riesgo": lam, "nº licitaciones": int(x.sum()),
                      "Σ compatibilidad": round(float(ctx["s"] @ x), 1),
                      "riesgo difuso": round(R.riesgo(x, ctx), 1),
                      "fitness": round(float(res["ranking"][0][1]), 1)})
    return pd.DataFrame(filas)
