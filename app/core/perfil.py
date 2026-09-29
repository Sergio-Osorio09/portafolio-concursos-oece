"""Profile Service: ficha técnica de la empresa.

En el sistema completo esta ficha la construye el chatbot (IA generativa). Aquí se llena con
un formulario o se elige una empresa simulada. La IA nunca calcula puntajes: solo entrega
estos datos estructurados al núcleo inteligente.
"""

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
