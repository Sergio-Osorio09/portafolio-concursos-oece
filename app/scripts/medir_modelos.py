"""Mide qué tan rápido responde cada modelo gratuito de OpenRouter con una entrevista real del asistente.

Uso (desde app/):  python scripts/medir_modelos.py
Lee OPENROUTER_API_KEY de .streamlit/secrets.toml o de la variable de entorno. La clave nunca se imprime.
"""
import os
import sys
import time
import tomllib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import asistente as A  # noqa: E402

MODELOS = [A.OPENROUTER_DEFECTO] + A.OPENROUTER_RESPALDO + ["nvidia/nemotron-3.5-lightning:free",
                                                            "google/gemma-4-31b-it:free"]

ruta = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".streamlit", "secrets.toml")
clave = os.environ.get("OPENROUTER_API_KEY", "")
if not clave and os.path.exists(ruta):
    with open(ruta, "rb") as f:
        clave = tomllib.load(f).get("OPENROUTER_API_KEY", "")
if not A.es_openrouter(clave):
    sys.exit("No hay una clave de OpenRouter (sk-or-...) en .streamlit/secrets.toml ni en OPENROUTER_API_KEY.")

historial = [{"role": "assistant", "content": "¿A qué se dedica tu empresa?"},
             {"role": "user", "content": "Somos una empresa de software y soporte técnico en Lima, con 5 años de "
                                         "experiencia, 6 personas y podemos llevar 3 proyectos a la vez."}]
print(f"{'modelo':45} {'segundos':>8}  resultado")
for m in dict.fromkeys(MODELOS):
    t = time.time()
    try:
        r = A.turno_recoleccion(clave, historial, {}, m)
        ok = f"OK ({r['modelo']}) · extrajo {len(r['campos'])} datos: {', '.join(r['campos'])}"
    except A.GeminiError as e:
        ok = f"ERROR: {e}"
    print(f"{m:45} {time.time() - t:8.1f}  {ok}")
