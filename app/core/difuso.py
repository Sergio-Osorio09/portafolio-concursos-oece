"""Motor de inferencia difusa Mamdani genérico.

Lo usan los DOS sistemas difusos del proyecto:
  - compatibilidad empresa-concurso (una evaluación por concurso)
  - riesgo del portafolio (una evaluación por combinación de concursos)

Pasos: fuzzificación (funciones triangulares/hombro) -> reglas SI-ENTONCES con AND = mínimo
-> implicación por mínimo -> agregación por máximo -> defuzzificación por centroide.
"""
import numpy as np


def trimf(x, a, b, c):
    """Triangular con hombros: (a,a,c) = hombro izquierdo, (a,c,c) = hombro derecho."""
    x = np.asarray(x, dtype=float)
    izq = (x - a) / (b - a) if b > a else np.ones_like(x)
    der = (c - x) / (c - b) if c > b else np.ones_like(x)
    return np.clip(np.minimum(izq, der), 0, 1)


class SistemaMamdani:
    def __init__(self, variables, salida, reglas, universo=(0, 100), puntos=201):
        self.variables = variables          # {var: {termino: (a, b, c)}}
        self.salida = salida                # {termino: (a, b, c)}
        self.reglas = reglas                # [({var: termino, ...}, termino_salida), ...]
        self.Y = np.linspace(*universo, puntos)
        self.mf_salida = {k: trimf(self.Y, *p) for k, p in salida.items()}

    def fuzzificar(self, entrada):
        return {v: {t: float(trimf(entrada[v], *p)) for t, p in terms.items()}
                for v, terms in self.variables.items()}

    def evaluar(self, entrada, detalle=False):
        """Devuelve el valor defuzzificado; con detalle=True también grados, reglas activas y la curva agregada."""
        grados = self.fuzzificar(entrada)
        agregada = np.zeros_like(self.Y)
        activas = []
        for ant, cons in self.reglas:
            w = min(grados[v][t] for v, t in ant.items())      # AND = mínimo
            if w > 0:
                agregada = np.maximum(agregada, np.minimum(w, self.mf_salida[cons]))  # implicación min, agregación max
                activas.append((ant, cons, w))
        valor = float((self.Y * agregada).sum() / agregada.sum()) if agregada.sum() > 0 else 0.0
        if not detalle:
            return valor
        activas.sort(key=lambda r: -r[2])
        return {"valor": valor, "grados": grados, "activas": activas, "agregada": agregada}


def regla_texto(ant, cons, salida="salida"):
    cond = " Y ".join(f"{v} es {t.upper()}" for v, t in ant.items())
    return f"SI {cond} ENTONCES {salida} es {cons.upper()}"
