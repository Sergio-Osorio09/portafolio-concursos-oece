"""Sistema difuso 2: riesgo del PORTAFOLIO (0-100), una evaluación por COMBINACIÓN de concursos.

El riesgo no se puede calcular concurso por concurso: dos concursos que solos son seguros pueden
ser riesgosos juntos (suman capital, personal y se cruzan en fechas). Por eso es un segundo
sistema difuso que recibe el portafolio completo y su salida entra al fitness del AG como penalidad.

Cambio respecto al notebook (pedido del profesor: riesgo por CANTIDAD de licitaciones):
  - `cantidad` = nº de licitaciones / capacidad operativa, ahora en [0, 2] (antes se cortaba en 1).
  - Nuevo término EXCESIVA (> capacidad operativa) con regla -> riesgo CRÍTICO.
  - El tope duro del AG sube a 2x la capacidad: ya no es una regla fija la que limita cuántas
    licitaciones se eligen, sino la lógica difusa (el "demasiadas" es gradual, no un corte exacto).
"""
import numpy as np

from .difuso import SistemaMamdani

VARIABLES = {
    "cantidad": {"baja": (0, 0, 0.5), "media": (0.25, 0.6, 0.95), "alta": (0.7, 1, 1.5), "excesiva": (1.05, 1.7, 1.7)},
    "capital":  {"holgado": (0, 0, 0.6), "ajustado": (0.4, 0.8, 1.2), "excedido": (1, 2, 2)},
    "personal": {"holgado": (0, 0, 0.7), "justo": (0.5, 0.9, 1.3), "excedido": (1, 2, 2)},
    "fechas":   {"bajo": (0, 0, 0.3), "medio": (0.15, 0.4, 0.7), "alto": (0.5, 1, 1)},
}
UNIVERSOS = {"cantidad": (0, 2), "capital": (0, 2), "personal": (0, 2), "fechas": (0, 1)}
DESCRIPCION = {
    "cantidad": "nº de licitaciones elegidas / capacidad operativa (proyectos simultáneos)",
    "capital": "monto total comprometido / capacidad financiera",
    "personal": "personal total requerido / personal disponible",
    "fechas": "choque promedio de cierres de postulación entre pares (preparar propuestas a la vez)",
}
SALIDA = {"bajo": (0, 0, 30), "medio": (15, 40, 65), "alto": (50, 70, 90), "critico": (80, 100, 100)}

# Base de reglas organizada como MATRIZ por nivel de cantidad:
#   cada nivel tiene un riesgo base y los demás factores lo ESCALAN un nivel.
#   Así la cantidad de licitaciones es el eje principal del riesgo (pedido del profesor) y ningún
#   rango de entradas queda sin reglas activas.
#
#                 base      +capital ajustado  +personal justo  +fechas medio  +fechas alto
#   baja          bajo      medio              medio            medio          medio
#   media         bajo      medio              medio            medio          alto
#   alta          medio     alto               alto             alto           crítico
#   excesiva      crítico   (ya es el máximo)
#   + globales:   capital excedido -> crítico ; personal excedido -> crítico
_ESCALA = {
    "baja":  {"base": "bajo",  "capital": "medio", "personal": "medio", "fechas_medio": "medio", "fechas_alto": "medio"},
    "media": {"base": "bajo",  "capital": "medio", "personal": "medio", "fechas_medio": "medio", "fechas_alto": "alto"},
    "alta":  {"base": "medio", "capital": "alto",  "personal": "alto",  "fechas_medio": "alto",  "fechas_alto": "critico"},
}
REGLAS = []
for _nivel, _c in _ESCALA.items():
    REGLAS += [
        ({"cantidad": _nivel}, _c["base"]),
        ({"cantidad": _nivel, "capital": "ajustado"}, _c["capital"]),
        ({"cantidad": _nivel, "personal": "justo"}, _c["personal"]),
        ({"cantidad": _nivel, "fechas": "medio"}, _c["fechas_medio"]),
        ({"cantidad": _nivel, "fechas": "alto"}, _c["fechas_alto"]),
    ]
REGLAS += [
    ({"cantidad": "excesiva"}, "critico"),     # más licitaciones que la capacidad operativa
    ({"capital": "excedido"}, "critico"),
    ({"personal": "excedido"}, "critico"),
]
MATRIZ = _ESCALA

SISTEMA = SistemaMamdani(VARIABLES, SALIDA, REGLAS)


def nivel(r):
    return "bajo" if r < 30 else "medio" if r < 55 else "alto" if r < 75 else "crítico"


def entradas(x, ctx):
    """x = cromosoma binario; ctx = datos del portafolio (montos, personal, conflictos, empresa)."""
    k = float(x.sum())
    pares = k * (k - 1) / 2
    return {
        "cantidad": min(2.0, k / ctx["capacidad_operativa"]),
        "capital": min(2.0, float(ctx["monto"] @ x) / ctx["capacidad_financiera"]),
        "personal": min(2.0, float(ctx["pers"] @ x) / ctx["personal_disponible"]),
        "fechas": min(1.0, float(x @ ctx["conf_sup"] @ x) / pares) if pares > 0 else 0.0,
    }


def riesgo(x, ctx, detalle=False):
    if x.sum() == 0:
        return {"valor": 0.0, "activas": [], "entradas": entradas(x, ctx)} if detalle else 0.0
    ent = entradas(x, ctx)
    if not detalle:
        return SISTEMA.evaluar(ent)
    r = SISTEMA.evaluar(ent, detalle=True)
    r["entradas"] = ent
    return r


def curva_por_cantidad(capacidad_operativa, capital_por=0.08, personal_por=0.1, fechas=0.1, k_max=None):
    """Riesgo vs nº de licitaciones: cada licitación agrega `capital_por` y `personal_por`
    (fracción de la capacidad de la empresa); el choque de fechas se mantiene fijo."""
    k_max = k_max or 2 * capacidad_operativa
    ks = np.arange(0, k_max + 1)
    vals = [SISTEMA.evaluar({"cantidad": min(2.0, k / capacidad_operativa), "capital": min(2.0, k * capital_por),
                             "personal": min(2.0, k * personal_por), "fechas": fechas if k > 1 else 0.0}) if k > 0 else 0.0 for k in ks]
    return ks, np.array(vals)
