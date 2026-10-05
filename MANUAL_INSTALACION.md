# Manual de instalación y puesta en marcha

Sistema inteligente de portafolio de concursos públicos (OECE) **con asistente conversacional Gemini**.

## Cómo funciona el flujo con el asistente

```
Usuario ⇄ Asistente Gemini  ──ficha──▶  Sistema inteligente  ──resultado──▶  Asistente Gemini ──▶ Usuario
          (te entrevista)               (filtro + difuso + AG)               (te lo explica)
```

1. **Entrevista.** Al abrir la app, el asistente te pregunta por tu empresa. Mientras conversas, a la derecha se va
   llenando la ficha (✅ / ⬜). El asistente solo extrae datos: **Python valida** rangos, tipos y departamentos, y decide
   cuándo la ficha está completa.
2. **Evaluación.** Cuando la ficha está completa y confirmas ("sí, evalúa") —o pulsas **▶ Evaluar**— los datos pasan al
   sistema inteligente (filtro duro → difuso de compatibilidad → algoritmo genético ⇄ difuso de riesgo).
3. **Interpretación.** El resultado vuelve al asistente, que lo explica en lenguaje sencillo (pestaña **💬 Asistente**) y
   responde tus preguntas de seguimiento usando solo esos datos. La IA **no calcula** puntajes ni elige concursos.

Si Gemini falla o no hay clave, la app no se rompe: puedes elegir una empresa simulada (sin IA) y la explicación se
muestra con el texto estándar del sistema.

---

## 1. Requisitos

| Requisito | Detalle |
|---|---|
| Python | 3.10 o superior (probado con 3.12). Descarga: <https://www.python.org/downloads/> |
| Internet | Para Gemini (siempre) y para la API/descarga del OECE (opcional: hay una muestra incluida) |
| Clave de API de Gemini | Gratuita, ver el paso 3 |
| Espacio | ~200 MB solo si descargas los datos completos del OECE (opcional) |

En Windows, al instalar Python marca **"Add python.exe to PATH"**.

## 2. Instalar el proyecto

Descomprime el proyecto (o clónalo) y abre una terminal en la carpeta `portafolio-concursos-oece-main`.

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
> Si PowerShell bloquea el script de activación, ejecuta una vez
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` y reintenta. En `cmd` usa `.venv\Scripts\activate.bat`.

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Obtener la clave de Gemini (gratis)

1. Entra a **Google AI Studio**: <https://aistudio.google.com/apikey> con tu cuenta de Google.
2. Pulsa **Crear clave de API** (no pide tarjeta de crédito para el nivel gratuito).
3. Copia la clave. **Trátala como una contraseña**: no la pegues en chats, capturas, correos ni en el código.

> ⚠️ **Si tu clave ya se compartió en algún lugar público o en una conversación, revócala** (en la misma página de AI
> Studio, elimínala y crea otra) y usa la nueva.

### Sobre el modelo (importante)
- Google anunció el **apagado de `gemini-2.5-flash` para el 16 de octubre de 2026** y ya restringe su uso a proyectos
  nuevos. Por eso la app usa por defecto el alias **`gemini-flash-latest`** (el mismo de tu comando `curl`), que
  siempre apunta al Flash vigente.
- Si un modelo responde "no disponible" (404), la app prueba automáticamente `gemini-3.5-flash` y luego
  `gemini-2.5-flash`.
- Puedes forzar uno concreto con `GEMINI_MODEL` (paso 4) o en la barra lateral → **🤖 Gemini → Modelo**.
- Los límites del nivel gratuito (peticiones por minuto y por día) los define Google y cambian con el tiempo; si ves el
  aviso de límite alcanzado, espera unos minutos. La app llama a Gemini solo cuando conversas, no en cada movimiento de
  un control.
- Nivel gratuito y privacidad: según los términos de Google para el uso gratuito, lo que envías puede usarse para mejorar
  sus productos (verifica los términos vigentes). No escribas datos confidenciales reales de terceros.

## 4. Configurar la clave (elige UNA forma)

**A) Archivo de secretos de Streamlit (recomendada)**
```bash
cd app
cp .streamlit/secrets.toml.example .streamlit/secrets.toml     # Windows: copy .streamlit\secrets.toml.example .streamlit\secrets.toml
```
Edita `app/.streamlit/secrets.toml` y reemplaza `pega-aqui-tu-clave` por tu clave. Este archivo ya está en `.gitignore`.

**B) Variable de entorno**
```powershell
# Windows PowerShell (solo esta terminal)
$env:GEMINI_API_KEY = "tu-clave"
```
```bash
# macOS / Linux
export GEMINI_API_KEY="tu-clave"
```

**C) Desde la interfaz**: barra lateral → **🤖 Gemini → Clave de API**. Solo dura mientras la sesión del navegador esté abierta.

## 5. Probar que todo funciona

Desde la carpeta `app`:
```bash
cd app
python scripts/prueba_asistente.py          # simulado: no usa internet ni clave. Debe terminar en "Todo OK ✔"
python scripts/prueba_asistente.py --real   # llama de verdad a Gemini: valida tu clave y el modelo
```
(Para `--real` define antes `GEMINI_API_KEY` como en la opción B.)

## 6. Ejecutar la aplicación
```bash
cd app
streamlit run app.py
```
Se abre en <http://localhost:8501>. La primera pantalla es el **asistente**: responde sus preguntas (puedes escribir
con naturalidad: "tengo 1.5 millones de capacidad, somos 6 personas…"). Al terminar verás los resultados y, en la
pestaña **💬 Asistente**, la explicación y un cuadro para hacerle preguntas.

Para usar una empresa simulada sin IA: barra lateral → **Empresa** → elige TechSolutions, Constructora o MedSupply.
Botón **↩️ Nueva conversación** para empezar de cero.

## 7. Datos de concursos (opcional)

| Opción | Cómo | Cuándo |
|---|---|---|
| Muestra de respaldo (incluida) | No hay que hacer nada | Probar de inmediato o sin internet |
| Datos completos del OECE | `python scripts/preparar_datos.py` (desde `app/`) | ~140 MB; crea el caché `app/data/oece_2026_procesos.parquet` |
| API en vivo | Barra lateral → *Fuente de concursos → API en vivo OECE* | Procesos publicados hoy |

## 8. Solución de problemas

| Síntoma | Causa y solución |
|---|---|
| "Falta la clave de API de Gemini" | Configura la clave (paso 4) y recarga la página. |
| "La clave de API de Gemini no es válida" | Copia otra vez la clave completa, sin espacios; si la revocaste, crea una nueva. |
| "no tiene permiso…" (403) | La clave está restringida o su proyecto no tiene habilitada la API de Gemini. Crea una clave nueva en AI Studio. |
| "Se alcanzó el límite de uso…" (429) | Cuota gratuita agotada (por minuto o por día). Espera un rato; la del día se reinicia diariamente. |
| "El modelo de Gemini no está disponible" (404) | Modelo retirado. Deja el modelo por defecto o prueba `gemini-3.5-flash`. |
| "No se pudo conectar con Gemini" | Sin internet, proxy o firewall bloqueando `generativelanguage.googleapis.com`. |
| `ModuleNotFoundError` | Activa el entorno virtual y repite `pip install -r requirements.txt`. |
| `streamlit` no se reconoce | Entorno virtual sin activar; o usa `python -m streamlit run app.py`. |
| Errores con `width="stretch"` | Streamlit antiguo: `pip install -U streamlit`. |
| El asistente pregunta algo que ya dijiste | Mira la ficha de la derecha: lo que no aparece con ✅ no quedó registrado. Repítelo con claridad. |
| "Ningún concurso pasa el filtro" | Amplía palabras clave, categorías o capacidad financiera en la pestaña **🏢 Empresa**. |

## 9. Seguridad de la clave

- El código **nunca** contiene la clave; se lee de `secrets.toml`, de `GEMINI_API_KEY` o de la barra lateral.
- `secrets.toml` y `.env` están en `.gitignore`. Antes de subir a GitHub, comprueba con `git status` que no aparezcan.
- Si la clave se filtra: elimínala en <https://aistudio.google.com/apikey> y crea otra.
