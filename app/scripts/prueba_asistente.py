"""Prueba del asistente Gemini, de punta a punta y SIN interfaz.

Uso (desde app/):
    python scripts/prueba_asistente.py           # simula las respuestas de Gemini: no usa internet ni clave
    python scripts/prueba_asistente.py --real    # llama de verdad a Gemini (requiere GEMINI_API_KEY)

El modo simulado verifica: la entrevista (extracción, validación y completitud de la ficha), el uso del
modelo de respaldo ante un 404, el manejo de errores (clave inválida, límite 429) y que el resultado del
sistema inteligente se convierte en el JSON que recibe el asistente para interpretarlo.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import asistente, compatibilidad, filtro, perfil, portafolio_ag as PA, riesgo as R  # noqa: E402


# ----------------------------------------------------------------------- Gemini simulado
class RespuestaFalsa:
    def __init__(self, codigo, cuerpo):
        self.status_code, self._cuerpo, self.text = codigo, cuerpo, json.dumps(cuerpo)

    def json(self):
        return self._cuerpo


def candidato(texto):
    return RespuestaFalsa(200, {"candidates": [{"content": {"parts": [{"text": texto}]}, "finishReason": "STOP"}]})


def turno(mensaje, actualizaciones=(), confirmado=False):
    return candidato(json.dumps({"mensaje": mensaje, "confirmado": confirmado, "actualizaciones": [
        {"campo": c, "valor": v} for c, v in actualizaciones]}))


GUION = [   # lo que "diría" Gemini en cada turno de la entrevista
    turno("¡Gracias! ¿Cuál es tu capacidad financiera y cuántas personas tienes?",
          [("rubro_keywords", "software; licencias; laptop; computadora; soporte tecnico"),
           ("categorias", "goods; services"), ("experiencia_anios", "5"), ("departamento_base", "Lima")]),
    turno("Perfecto. ¿Cobertura nacional o regional?",
          [("capacidad_financiera", "1500000"), ("monto_minimo_interes", "20000"),
           ("presupuesto_postulaciones", "120000"), ("personal_disponible", "6"),
           ("capacidad_operativa", "99")]),        # 99 está fuera de rango: debe ajustarse a 20 y avisar
    turno("Resumen listo. ¿Evaluamos?", [("cobertura", "nacional"), ("capacidad_operativa", "3")]),
    turno("¡Vamos a evaluar!", [], confirmado=True),
]


class GeminiFalso:
    def __init__(self):
        self.llamadas, self.turnos = [], iter(GUION)

    def post(self, url, headers=None, json=None, timeout=None):      # noqa: A002 (misma firma que requests)
        self.llamadas.append({"url": url, "headers": headers, "cuerpo": json})
        sistema = json["systemInstruction"]["parts"][0]["text"]
        if "gemini-2.5-flash-agotado" in url:
            return RespuestaFalsa(404, {"error": {"message": "This model is no longer available."}})
        if "ESTADO ACTUAL" in sistema:
            return next(self.turnos)
        return candidato("**Resumen:** se recomienda un portafolio con buen equilibrio entre compatibilidad y riesgo.")


def verificar(cond, msg):
    print(("  ✔ " if cond else "  ✘ ") + msg)
    if not cond:
        sys.exit(1)


# ----------------------------------------------------------------------- datos sintéticos
def concursos_sinteticos(fecha_ref, n=45):
    rng = np.random.default_rng(7)
    temas = ["adquisicion de laptops y computadoras", "licencias de software ofimatico", "servicio de soporte tecnico",
             "equipos de computo e impresoras", "sistema informatico de gestion", "servidores y firewall",
             "obra de pavimentacion de vereda", "adquisicion de medicamentos", "servicio de limpieza"]
    deps = ["LIMA", "LIMA", "CALLAO", "JUNIN", "AREQUIPA", "PIURA", "CUSCO"]
    filas = []
    for i in range(n):
        tema = temas[i % len(temas)]
        filas.append({
            "ocid": f"ocds-test-{i:04d}", "titulo": tema, "descripcion": f"{tema} para la entidad {i}",
            "texto_items": tema, "entidad": f"MUNICIPALIDAD DISTRITAL {i}", "departamento": deps[i % len(deps)],
            "categoria": "works" if "obra" in tema else ("goods" if i % 2 else "services"),
            "metodo": "Adjudicación Simplificada", "estado": None, "estado_detalle": "CONVOCADO",
            "monto": float(rng.integers(30_000, 1_300_000)), "moneda": "PEN",
            "cierre_est": fecha_ref + pd.Timedelta(days=int(rng.integers(3, 26))),
            "duracion_dias": float(rng.integers(30, 150)), "fecha_publicacion": fecha_ref})
    return pd.DataFrame(filas)


def principal():
    sys.modules["time"].sleep = lambda s: None            # los reintentos no deben esperar en la prueba
    falso = GeminiFalso()
    asistente.requests.post = falso.post

    print("1. Entrevista (Gemini simulado)")
    msgs = [{"role": "assistant", "content": "Hola, ¿a qué se dedica tu empresa?"}]
    campos, ultimo = {}, None
    for texto_usuario in ["Vendemos laptops y software, 5 años en Lima", "Tengo 1.5 millones, 6 personas",
                          "Nacional", "sí, evalúa"]:
        msgs.append({"role": "user", "content": texto_usuario})
        ultimo = asistente.turno_recoleccion("CLAVE-DE-PRUEBA", msgs, campos)
        msgs.append({"role": "assistant", "content": ultimo["mensaje"]})
        campos = ultimo["campos"]
        print(f"   usuario: {texto_usuario!r} → faltan {len(ultimo['faltan'])} · confirmado={ultimo['confirmado']}")
    primera = falso.llamadas[0]
    verificar(primera["headers"]["x-goog-api-key"] == "CLAVE-DE-PRUEBA", "la clave viaja en el encabezado x-goog-api-key")
    verificar("gemini-flash-latest:generateContent" in primera["url"], "usa el modelo por defecto configurado")
    verificar(primera["cuerpo"]["contents"][0]["role"] == "user", "el historial enviado empieza con el usuario")
    verificar("CLAVE-DE-PRUEBA" not in json.dumps(primera["cuerpo"]), "la clave no se filtra al contenido del prompt")
    verificar(campos["capacidad_operativa"] == 3, "el valor corregido por el usuario reemplaza al anterior")
    verificar(any("fuera del rango" in m["content"] for m in msgs if m["role"] == "assistant"),
              "un valor fuera de rango se ajusta y se avisa al usuario")
    verificar(perfil.faltantes(campos) == [] and ultimo["confirmado"], "ficha completa + confirmación del usuario → evaluar")
    ficha = perfil.ficha_desde_campos(campos)
    verificar(perfil.validar(ficha) == [] and ficha["cobertura"] == "nacional", "la ficha final es válida")

    print("2. Confirmación prematura (el modelo dice 'confirmado' con datos faltantes)")
    falso.turnos = iter([turno("¡Evaluemos!", [("categorias", "goods")], confirmado=True)])
    r = asistente.turno_recoleccion("K", [{"role": "user", "content": "hola"}], {})
    verificar(r["confirmado"] is False and len(r["faltan"]) > 0, "Python ignora la confirmación si faltan datos")

    print("3. Modelo retirado (404) → modelo de respaldo")
    antes = len(falso.llamadas)
    texto, usado = asistente.llamar_gemini("K", [{"role": "user", "parts": [{"text": "hola"}]}], "s",
                                           modelo="gemini-2.5-flash-agotado")
    verificar(usado == asistente.MODELOS_RESPALDO[0] and len(falso.llamadas) - antes == 2,
              f"tras el 404 se usó el respaldo «{usado}»")

    print("4. Errores")
    asistente.requests.post = lambda *a, **k: RespuestaFalsa(400, {"error": {"message": "API key not valid. Please pass a valid API key."}})
    try:
        asistente.llamar_gemini("mala", [{"role": "user", "parts": [{"text": "x"}]}], "s")
        verificar(False, "debía fallar")
    except asistente.GeminiError as e:
        verificar("no es válida" in str(e), f"clave inválida → «{e}»")
    asistente.requests.post = lambda *a, **k: RespuestaFalsa(429, {"error": {"message": "quota"}})
    try:
        asistente.llamar_gemini("k", [{"role": "user", "parts": [{"text": "x"}]}], "s")
        verificar(False, "debía fallar")
    except asistente.GeminiError as e:
        verificar("límite" in str(e), f"límite gratuito (429) → «{e}»")
    try:
        asistente.llamar_gemini("", [], "s")
        verificar(False, "debía fallar")
    except asistente.GeminiError as e:
        verificar(e.codigo == "sin_clave", "sin clave → mensaje claro, sin llamar a la red")
    asistente.requests.post = falso.post

    print("5. Sistema inteligente + interpretación (datos sintéticos)")
    fecha_ref = pd.Timestamp("2026-09-20")
    vig = concursos_sinteticos(fecha_ref)
    fac, embudo = filtro.aplicar(vig, ficha)
    ev = compatibilidad.evaluar(fac, ficha, fecha_ref)
    cand = ev[ev.score >= 40].head(12).drop(columns=["regla_dominante"]).reset_index(drop=True)
    cand.index = [f"C{i + 1}" for i in range(len(cand))]
    ctx = PA.construir_contexto(cand, ficha, fecha_ref)
    lambdas = {"riesgo": 1.0, "recursos": 1.0, "fechas": 0.3, "ventana": 0.2}
    fit = PA.Evaluador(ctx, lambdas, "descuento", False)
    res = PA.algoritmo_genetico(ctx, fit)
    xb = res["ranking"][0][0]
    bar = PA.barrido_lambda(ctx, lambdas, [0, 1, 2], "descuento", False)
    print(f"   factibles={len(fac)} candidatos={len(cand)} elegidas={int(xb.sum())} riesgo={R.riesgo(xb, ctx):.1f}")
    resultados = asistente.contexto_resultados(ficha, cand, xb, res["ranking"], ctx, fit, embudo, bar, 1.0, "descuento")
    texto_json = json.dumps(resultados, ensure_ascii=False)
    verificar(len(resultados["portafolio_recomendado"]["concursos"]) == int(xb.sum()), "el JSON lista los concursos elegidos")
    verificar(all(c["id"] in cand.index for c in resultados["portafolio_recomendado"]["concursos"]), "los IDs coinciden con los candidatos")
    verificar(len(texto_json) < 20_000, f"el JSON es compacto ({len(texto_json):,} caracteres)")
    explicacion, usado = asistente.interpretar("K", resultados)
    verificar(explicacion.startswith("**Resumen"), "el asistente devuelve la interpretación")
    sistema = falso.llamadas[-1]["cuerpo"]["systemInstruction"]["parts"][0]["text"]
    verificar("RESULTADOS (JSON)" in sistema and resultados["portafolio_recomendado"]["concursos"][0]["id"] in sistema,
              "el prompt de interpretación contiene los resultados del sistema")
    seguimiento, _ = asistente.interpretar("K", resultados, historial=[
        {"role": "assistant", "content": explicacion}, {"role": "user", "content": "¿Por qué no postulo a C9?"}])
    roles = [c["role"] for c in falso.llamadas[-1]["cuerpo"]["contents"]]
    verificar(roles == ["user", "model", "user"], f"el seguimiento alterna roles correctamente {roles}")
    print("\nTodo OK ✔")


def prueba_real():
    clave = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not clave:
        sys.exit("Define la variable de entorno GEMINI_API_KEY para usar --real (ver MANUAL_INSTALACION.md).")
    modelo = os.environ.get("GEMINI_MODEL") or asistente.MODELO_DEFECTO
    print(f"Llamando a Gemini (modelo pedido: {modelo})…")
    t0 = time.time()
    r = asistente.turno_recoleccion(clave, [{"role": "user", "content":
                                    "Hola, vendo laptops y software a entidades públicas, tengo 5 años de experiencia."}], {}, modelo)
    print(f"Modelo que respondió: {r['modelo']} ({time.time() - t0:.1f} s)\n")
    print("Asistente:", r["mensaje"])
    print("Datos extraídos:", json.dumps(r["nuevos"], ensure_ascii=False))
    print("Aún faltan:", [perfil.ETIQUETAS.get(f, f) for f in r["faltan"]])


if __name__ == "__main__":
    try:
        prueba_real() if "--real" in sys.argv else principal()
    except asistente.GeminiError as e:
        sys.exit(f"Error de Gemini: {e}")
