"""Profile Service: ficha técnica de la empresa.

La ficha la construye el asistente conversacional (Gemini, ver `asistente.py`), o se llena con un
formulario, o se elige una empresa simulada. La IA nunca calcula puntajes: solo entrega estos datos
estructurados al núcleo inteligente. Este módulo es quien VALIDA y NORMALIZA lo que dice la IA
(rangos, tipos, departamentos), de modo que nada que invente o escriba mal el modelo llega al sistema.
"""
import math
import re
import unicodedata

EMPRESAS = {
    "TechSolutions SAC (TI)": {
        "rubro_keywords": [
            "software", "licencias", "computadora", "computadoras", "computo", "laptop", "laptops",
            "servidor", "servidores", "impresora", "impresoras", "informatica", "informaticos",
            "hardware", "red de datos", "cableado estructurado", "switch", "firewall",
            "tecnologia de la informacion", "soporte tecnico", "aplicativo", "plataforma web",
            "sistema informatico", "equipos de computo", "data center", "ciberseguridad",
            "equipos informaticos", "toner", "sistema de informacion",
        ],
        "categorias": ["goods", "services"],
        "capacidad_financiera": 1_500_000,
        "monto_minimo_interes": 20_000,
        "presupuesto_postulaciones": 120_000,
        "experiencia_anios": 5,
        "personal_disponible": 6,
        "capacidad_operativa": 3,
        "departamento_base": "LIMA",
        "cobertura": "nacional",
        "departamentos_cobertura": ["LIMA", "CALLAO", "ICA", "JUNIN"],
        "ventana_disponible": ("2026-09-01", "2027-03-31"),
    },
    "Constructora Andina EIRL (obras)": {
        "rubro_keywords": [
            "obra", "construccion", "mejoramiento", "rehabilitacion", "pavimentacion", "vereda",
            "veredas", "pista", "pistas", "puente", "saneamiento", "agua potable", "alcantarillado",
            "infraestructura", "edificacion", "losa deportiva", "muro de contencion", "carretera",
            "camino vecinal", "drenaje", "institucion educativa", "reparacion",
        ],
        "categorias": ["works"],
        "capacidad_financiera": 3_000_000,
        "monto_minimo_interes": 100_000,
        "presupuesto_postulaciones": 250_000,
        "experiencia_anios": 8,
        "personal_disponible": 25,
        "capacidad_operativa": 3,
        "departamento_base": "JUNIN",
        "cobertura": "regional",
        "departamentos_cobertura": ["JUNIN", "PASCO", "HUANCAVELICA", "LIMA", "AYACUCHO", "HUANUCO"],
        "ventana_disponible": ("2026-09-01", "2027-06-30"),
    },
    "MedSupply Peru SAC (insumos médicos)": {
        "rubro_keywords": [
            "medicamento", "medicamentos", "material medico", "insumos medicos", "dispositivos medicos",
            "reactivos", "laboratorio", "jeringa", "jeringas", "guantes", "mascarilla", "oxigeno",
            "equipo medico", "equipamiento hospitalario", "farmaceutico", "farmaceuticos",
            "material quirurgico", "sutura", "gasa", "catéter", "cateter", "biomedico", "odontologico",
        ],
        "categorias": ["goods"],
        "capacidad_financiera": 800_000,
        "monto_minimo_interes": 15_000,
        "presupuesto_postulaciones": 60_000,
        "experiencia_anios": 3,
        "personal_disponible": 4,
        "capacidad_operativa": 4,
        "departamento_base": "LIMA",
        "cobertura": "nacional",
        "departamentos_cobertura": ["LIMA", "CALLAO", "LA LIBERTAD", "AREQUIPA", "PIURA"],
        "ventana_disponible": ("2026-09-01", "2027-03-31"),
    },
}

CATEGORIAS = {"goods": "Bienes", "services": "Servicios", "works": "Obras"}


def validar(ficha):
    """Normaliza/valida la ficha (lo que haría el Profile Service tras el chatbot)."""
    errores = []
    if not ficha["rubro_keywords"]:
        errores.append("Debe indicar al menos una palabra clave del rubro.")
    if not ficha["categorias"]:
        errores.append("Debe elegir al menos una categoría.")
    if ficha["monto_minimo_interes"] >= ficha["capacidad_financiera"]:
        errores.append("El monto mínimo de interés debe ser menor que la capacidad financiera.")
    if ficha["capacidad_operativa"] < 1 or ficha["personal_disponible"] < 1:
        errores.append("Capacidad operativa y personal deben ser al menos 1.")
    return errores


# ====================== Utilidades para la ficha armada por el asistente ======================
DEPARTAMENTOS = [
    "AMAZONAS", "ANCASH", "APURIMAC", "AREQUIPA", "AYACUCHO", "CAJAMARCA", "CALLAO", "CUSCO",
    "HUANCAVELICA", "HUANUCO", "ICA", "JUNIN", "LA LIBERTAD", "LAMBAYEQUE", "LIMA", "LORETO",
    "MADRE DE DIOS", "MOQUEGUA", "PASCO", "PIURA", "PUNO", "SAN MARTIN", "TACNA", "TUMBES", "UCAYALI",
]
_ALIAS_DEP = {"CUZCO": "CUSCO", "PROVINCIA CONSTITUCIONAL DEL CALLAO": "CALLAO", "LIMA METROPOLITANA": "LIMA",
              "LIMA PROVINCIAS": "LIMA", "HUANUCO": "HUANUCO",
              # ciudades frecuentes -> departamento (el usuario suele decir la ciudad)
              "IQUITOS": "LORETO", "TRUJILLO": "LA LIBERTAD", "CHICLAYO": "LAMBAYEQUE", "HUANCAYO": "JUNIN",
              "PUCALLPA": "UCAYALI", "TARAPOTO": "SAN MARTIN", "MOYOBAMBA": "SAN MARTIN", "HUARAZ": "ANCASH",
              "CHIMBOTE": "ANCASH", "JULIACA": "PUNO", "PUERTO MALDONADO": "MADRE DE DIOS", "ILO": "MOQUEGUA",
              "CHACHAPOYAS": "AMAZONAS", "BAGUA": "AMAZONAS", "ABANCAY": "APURIMAC", "ANDAHUAYLAS": "APURIMAC",
              "HUAMANGA": "AYACUCHO", "CERRO DE PASCO": "PASCO", "SULLANA": "PIURA", "TALARA": "PIURA",
              "CHINCHA": "ICA", "PISCO": "ICA", "NAZCA": "ICA", "TINGO MARIA": "HUANUCO", "JAEN": "CAJAMARCA",
              "CANETE": "LIMA", "HUACHO": "LIMA", "TUMBES": "TUMBES", "TACNA": "TACNA"}
_ALIAS_CAT = {"goods": "goods", "bienes": "goods", "bien": "goods",
              "services": "services", "servicios": "services", "servicio": "services",
              "works": "works", "obras": "works", "obra": "works"}

VENTANA_DEFECTO = ("2026-09-01", "2027-06-30")   # solo se usa con las penalidades clásicas (opcionales)

# mismos rangos que los campos numéricos del formulario de la interfaz
LIMITES = {
    "capacidad_financiera": (10_000, 50_000_000),
    "monto_minimo_interes": (0, 5_000_000),
    "presupuesto_postulaciones": (5_000, 5_000_000),
    "experiencia_anios": (0, 60),
    "personal_disponible": (1, 500),
    "capacidad_operativa": (1, 20),
}
ETIQUETAS = {
    "rubro_keywords": "Rubro (palabras clave)",
    "categorias": "Categorías que atiende",
    "capacidad_financiera": "Capacidad financiera (S/)",
    "monto_minimo_interes": "Monto mínimo de interés (S/)",
    "presupuesto_postulaciones": "Presupuesto de postulaciones (S/)",
    "experiencia_anios": "Años de experiencia",
    "personal_disponible": "Personal disponible",
    "capacidad_operativa": "Proyectos simultáneos que soporta",
    "departamento_base": "Departamento base",
    "cobertura": "Cobertura",
    "departamentos_cobertura": "Departamentos de cobertura",
}
REQUERIDOS = ["rubro_keywords", "categorias", "capacidad_financiera", "monto_minimo_interes",
              "presupuesto_postulaciones", "experiencia_anios", "personal_disponible",
              "capacidad_operativa", "departamento_base", "cobertura"]
DEFECTOS = {"rubro_keywords": [], "categorias": [], "capacidad_financiera": 1_000_000,
            "monto_minimo_interes": 20_000, "presupuesto_postulaciones": 100_000, "experiencia_anios": 3,
            "personal_disponible": 5, "capacidad_operativa": 3, "departamento_base": "LIMA",
            "cobertura": "nacional"}


def _norm(txt):
    t = unicodedata.normalize("NFKD", str(txt or "")).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", t.lower()).strip()


def _num(v):
    """Número desde int/float/texto ('1500000', '1,500,000', 'S/ 800 000'). None si no se puede."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        x = float(v)
    elif isinstance(v, str):
        t = re.sub(r"[^\d.,\-]", "", v.replace(" ", ""))
        if not t:
            return None
        if "," in t and "." in t:
            t = t.replace(",", "")
        elif t.count(",") > 1:
            t = t.replace(",", "")
        elif t.count(".") > 1:
            t = t.replace(".", "")
        elif "," in t:
            t = t.replace(",", "") if len(t.split(",")[1]) == 3 else t.replace(",", ".")
        try:
            x = float(t)
        except ValueError:
            return None
    else:
        return None
    return x if math.isfinite(x) else None


def _lista(v):
    if isinstance(v, str):
        return [p for p in re.split(r"[,;\n]", v) if p.strip()]
    return list(v) if isinstance(v, (list, tuple)) else []


def _depto(txt):
    d = _norm(txt).upper()
    d = _ALIAS_DEP.get(d, d)
    if d in DEPARTAMENTOS:
        return d
    for nombre in sorted(DEPARTAMENTOS + list(_ALIAS_DEP), key=len, reverse=True):   # "departamento de Lima"
        if re.search(r"\b" + re.escape(nombre) + r"\b", d):
            return _ALIAS_DEP.get(nombre, nombre)
    return None


def limpiar_campos(crudo):
    """Valida y normaliza campos sueltos (lo que extrajo la IA de la conversación).

    Devuelve (campos_validos, avisos). Lo que no se reconoce o no se puede interpretar se descarta;
    un número fuera de rango se ajusta al límite y se avisa. Un 0 en un campo cuyo mínimo es > 0 se
    toma como "no informado" (los modelos de lenguaje a veces rellenan con 0)."""
    campos, avisos = {}, []
    if not isinstance(crudo, dict):
        return campos, avisos

    nombre = crudo.get("nombre")
    if isinstance(nombre, str) and nombre.strip():
        campos["nombre"] = nombre.strip()[:80]

    kws = []
    for w in _lista(crudo.get("rubro_keywords")):
        w = str(w).strip().lower()
        if w and w not in kws:
            kws.append(w)
    if kws:
        campos["rubro_keywords"] = kws[:40]

    cats = []
    for c in _lista(crudo.get("categorias")):
        c = _ALIAS_CAT.get(_norm(c))
        if c and c not in cats:
            cats.append(c)
    if cats:
        campos["categorias"] = cats

    for campo, (lo, hi) in LIMITES.items():
        if campo not in crudo:
            continue
        n = _num(crudo[campo])
        if n is None or (n <= 0 and lo > 0):
            continue
        ajustado = min(max(int(round(n)), lo), hi)
        if ajustado != int(round(n)):
            avisos.append(f"{ETIQUETAS[campo]}: {int(round(n)):,} está fuera del rango permitido "
                          f"({lo:,} a {hi:,}); lo ajusté a {ajustado:,}.")
        campos[campo] = ajustado

    if crudo.get("departamento_base"):
        d = _depto(crudo["departamento_base"])
        if d:
            campos["departamento_base"] = d
        else:
            avisos.append(f"No reconozco el departamento «{crudo['departamento_base']}».")

    cob = _norm(crudo.get("cobertura"))
    if cob.startswith("nac"):
        campos["cobertura"] = "nacional"
    elif cob.startswith("reg"):
        campos["cobertura"] = "regional"

    deps = []
    for t in _lista(crudo.get("departamentos_cobertura")):
        d = _depto(t)
        if d is None:
            avisos.append(f"No reconozco el departamento «{str(t).strip()}».")
        elif d not in deps:
            deps.append(d)
    if deps:
        campos["departamentos_cobertura"] = deps
    return campos, avisos


def faltantes(campos):
    """Campos obligatorios que aún no se han recogido (decisión determinística, no de la IA)."""
    f = [c for c in REQUERIDOS if c not in campos]
    if campos.get("cobertura") == "regional" and not campos.get("departamentos_cobertura"):
        f.append("departamentos_cobertura")
    return f


def ficha_desde_campos(campos):
    """Arma la ficha completa que consume el núcleo inteligente (mismas claves que EMPRESAS)."""
    c = {**DEFECTOS, **{k: v for k, v in campos.items() if v not in (None, [], "")}}
    deps = list(c.get("departamentos_cobertura") or [])
    if c["departamento_base"] not in deps:
        deps = [c["departamento_base"]] + deps
    return {
        "nombre": c.get("nombre") or "Mi empresa",
        "rubro_keywords": list(c["rubro_keywords"]), "categorias": list(c["categorias"]),
        "capacidad_financiera": int(c["capacidad_financiera"]),
        "monto_minimo_interes": int(c["monto_minimo_interes"]),
        "presupuesto_postulaciones": int(c["presupuesto_postulaciones"]),
        "experiencia_anios": int(c["experiencia_anios"]), "personal_disponible": int(c["personal_disponible"]),
        "capacidad_operativa": int(c["capacidad_operativa"]),
        "departamento_base": c["departamento_base"], "cobertura": c["cobertura"],
        "departamentos_cobertura": deps, "ventana_disponible": VENTANA_DEFECTO,
    }


def valor_texto(campo, v):
    """Representación legible de un campo para mostrarlo al usuario."""
    if isinstance(v, list):
        if campo == "categorias":
            return ", ".join(CATEGORIAS.get(x, x) for x in v)
        return ", ".join(v[:8]) + ("…" if len(v) > 8 else "")
    if campo in ("capacidad_financiera", "monto_minimo_interes", "presupuesto_postulaciones"):
        return f"S/ {int(v):,}"
    return str(v).capitalize() if campo == "cobertura" else str(v)
