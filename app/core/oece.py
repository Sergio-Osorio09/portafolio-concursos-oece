"""Adaptador OECE + normalizador de concursos.

Dos fuentes oficiales, mismo formato OCDS:
  1. Descarga masiva (bulk) publicada por el OECE en el Registro de Open Contracting
     (corte mensual, ~140 MB). Se procesa UNA vez y se guarda en caché (parquet).
  2. API en vivo del portal de Contrataciones Abiertas del OECE
     (https://contratacionesabiertas.oece.gob.pe/api/v1/releases), paginada de 20 en 20,
     los releases más recientes primero.

El resto del sistema solo ve el DataFrame normalizado que sale de `extraer()`:
así el núcleo inteligente no queda acoplado a la API externa (decisión 17 del documento).
"""
import gzip
import json
import os
import re
import unicodedata

import pandas as pd
import requests

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ANIO = 2026
URL_BULK = f"https://data.open-contracting.org/es/publication/135/download?name={ANIO}.jsonl.gz"
URL_API = "https://contratacionesabiertas.oece.gob.pe/api/v1/releases"
ARCHIVO_BULK = os.path.join(DATA_DIR, f"oece_{ANIO}.jsonl.gz")
CACHE = os.path.join(DATA_DIR, f"oece_{ANIO}_procesos.parquet")
MUESTRA = os.path.join(DATA_DIR, "muestra_respaldo.parquet")

FECHAS = ["inicio_postulacion", "fin_postulacion", "fin_consultas", "fecha_publicacion"]
DIAS_PRESENTACION = 8     # días típicos entre fin de consultas y presentación de ofertas (supuesto)
DIAS_SIN_CONSULTAS = 15   # si no hay etapa de consultas: días desde la publicación (supuesto)

try:
    import orjson
    loads = orjson.loads
except ImportError:
    loads = json.loads


def norm(txt):
    txt = unicodedata.normalize("NFKD", str(txt or "")).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", txt.lower()).strip()


def departamento(rel):
    buyer_id = (rel.get("buyer") or {}).get("id")
    for p in rel.get("parties") or []:
        if p.get("id") == buyer_id or "buyer" in (p.get("roles") or []):
            a = p.get("address") or {}
            for k in ("department", "region", "locality"):
                v = a.get(k)
                if isinstance(v, dict):
                    v = v.get("name") or v.get("description")
                if v:
                    return norm(v).upper()
    return "DESCONOCIDO"


def extraer(rel):
    """Release OCDS -> fila plana con los campos que usa el sistema."""
    t = rel.get("tender") or {}
    val = t.get("value") or {}
    tp = t.get("tenderPeriod") or {}
    ep = t.get("enquiryPeriod") or {}
    cp = t.get("contractPeriod") or {}
    items = t.get("items") or []
    txt_items = " ".join(
        f"{i.get('description', '')} {(i.get('classification') or {}).get('description', '')}"
        for i in items[:30]
    )
    return {
        "ocid": rel.get("ocid"),
        "titulo": (t.get("title") or "")[:300],
        "descripcion": (t.get("description") or "")[:500],
        "texto_items": txt_items[:1500],
        "entidad": (rel.get("buyer") or {}).get("name", ""),
        "departamento": departamento(rel),
        "categoria": t.get("mainProcurementCategory"),
        "metodo": t.get("procurementMethodDetails") or t.get("procurementMethod"),
        "estado": t.get("status"),
        # el OECE no llena tender.status: el estado real viene en items[].statusDetails
        "estado_detalle": (items[0].get("statusDetails") if items else None),
        "monto": val.get("amount"),
        "moneda": val.get("currency"),
        "inicio_postulacion": tp.get("startDate"),
        "fin_postulacion": tp.get("endDate"),
        "fin_consultas": ep.get("endDate"),
        "duracion_dias": cp.get("durationInDays"),
        "fecha_publicacion": t.get("datePublished") or rel.get("date"),
    }


def _tipar(df):
    for c in FECHAS:
        df[c] = pd.to_datetime(df[c], errors="coerce", utc=True).dt.tz_convert(None)
    df["monto"] = pd.to_numeric(df["monto"], errors="coerce")
    df["duracion_dias"] = pd.to_numeric(df["duracion_dias"], errors="coerce")
    # En los datos actuales del OECE tenderPeriod trae inicio = fin (solo la fecha de convocatoria),
    # así que el cierre de postulación se ESTIMA desde el fin de la etapa de consultas.
    est = df["fin_consultas"].dt.normalize() + pd.Timedelta(days=DIAS_PRESENTACION)
    est = est.fillna(df["fecha_publicacion"].dt.normalize() + pd.Timedelta(days=DIAS_SIN_CONSULTAS))
    real = df["fin_postulacion"].where(df["fin_postulacion"] > df["inicio_postulacion"])
    df["cierre_est"] = real.fillna(est)
    return df


# ---------------- Fuente 1: descarga masiva ----------------
def descargar_bulk(progreso=None):
    if os.path.exists(ARCHIVO_BULK) and os.path.getsize(ARCHIVO_BULK) > 0:
        return ARCHIVO_BULK
    os.makedirs(DATA_DIR, exist_ok=True)
    with requests.get(URL_BULK, stream=True, timeout=120) as r:
        r.raise_for_status()
        total, hecho = int(r.headers.get("content-length", 0)), 0
        with open(ARCHIVO_BULK, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                hecho += len(chunk)
                if progreso and total:
                    progreso(hecho / total)
    return ARCHIVO_BULK


def procesar_bulk():
    filas = []
    with gzip.open(ARCHIVO_BULK, "rt", encoding="utf-8") as f:
        for linea in f:
            try:
                obj = loads(linea)
            except Exception:
                continue
            filas.append(extraer(obj.get("compiledRelease", obj)))
    df = _tipar(pd.DataFrame(filas))
    df.to_parquet(CACHE, index=False)
    return df


def cargar_bulk():
    """Lee el caché; si no existe, usa la muestra de respaldo incluida en el proyecto."""
    if os.path.exists(CACHE):
        return pd.read_parquet(CACHE), "Descarga masiva OECE (caché local)"
    if os.path.exists(MUESTRA):
        return pd.read_parquet(MUESTRA), "Muestra de respaldo (sin conexión)"
    raise FileNotFoundError("Ejecute scripts/preparar_datos.py para descargar los datos del OECE.")


# ---------------- Fuente 2: API en vivo ----------------
def consultar_api(paginas=10, progreso=None):
    """Trae los releases más recientes de la API oficial (20 por página)."""
    filas = []
    for p in range(1, paginas + 1):
        r = requests.get(URL_API, params={"format": "json", "page": p}, timeout=30)
        r.raise_for_status()
        for rel in r.json().get("releases", []):
            filas.append(extraer(rel))
        if progreso:
            progreso(p / paginas)
    df = _tipar(pd.DataFrame(filas))
    # la API devuelve un release por evento: se queda el más reciente de cada proceso
    return df.sort_values("fecha_publicacion").drop_duplicates("ocid", keep="last")


ESTADOS_ADJUDICADOS = ["ADJUDICADO", "CONSENTIDO", "CONTRATADO"]


def tabla_montos_historicos(df):
    """Mediana del monto de procesos YA adjudicados, por categoría y método.
    Se usa para estimar el monto de los convocados cuyo valor estimado está reservado
    (la Ley 32069 permite reservarlo hasta el otorgamiento de la buena pro)."""
    h = df[df["estado_detalle"].isin(ESTADOS_ADJUDICADOS) & (df["monto"] > 0) & (df["moneda"].fillna("PEN") == "PEN")]
    por_metodo = h.groupby(["categoria", "metodo"])["monto"].agg(["median", "size"])
    por_metodo = por_metodo[por_metodo["size"] >= 20]["median"]
    por_cat = h.groupby("categoria")["monto"].median()
    return {"metodo": por_metodo.to_dict(), "categoria": por_cat.to_dict()}


def concursos_vigentes(df, fecha_ref=None, min_vigentes=50, max_retro=180):
    """Vigente = el proceso sigue CONVOCADO (aún no adjudicado/desierto/nulo) y su cierre
    estimado de postulación no ha pasado en la fecha de referencia.
    Para la descarga masiva (corte mensual) la referencia es la fecha de corte; si hay pocos
    vigentes se retrocede de 7 en 7 días (simula consultar el sistema en ese momento)."""
    corte = df["fecha_publicacion"].max().normalize()
    base = fecha_ref if fecha_ref is not None else min(pd.Timestamp.today().normalize(), corte)
    convocado = df["estado_detalle"].fillna("").str.upper().str.startswith("CONVOCADO")

    def abiertos(f):
        return df[convocado & (df["cierre_est"] >= f) & (df["fecha_publicacion"] <= f + pd.Timedelta(days=1))]

    retro, vig = 0, abiertos(base)
    while len(vig) < min_vigentes and retro < max_retro:
        retro += 7
        vig = abiertos(base - pd.Timedelta(days=retro))
    return vig.copy(), base - pd.Timedelta(days=retro), corte, retro
