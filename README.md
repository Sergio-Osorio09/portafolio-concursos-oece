# Sistema Inteligente de Portafolio de Concursos Públicos (OECE)

**Asistente conversacional (Gemini) + lógica difusa + algoritmo genético con riesgo difuso en el fitness**, aplicado a los
datos abiertos oficiales de contrataciones públicas del Perú (OECE / SEACE, formato OCDS).

Proyecto del curso **Software Inteligente**.

## ¿Qué hace y cómo trabaja realmente?

El flujo tiene tres etapas. Solo la del medio calcula; la IA generativa **nunca** puntúa ni elige concursos.

```
Usuario ⇄ Asistente Gemini ──ficha──▶ Sistema inteligente (sin IA) ──resultado──▶ Asistente Gemini ──▶ Usuario
          (te entrevista)             filtro → difuso #1 → AG ⇄ difuso #2          (te lo explica)
```

1. **Entrevista (Gemini).** Al abrir la app, el asistente conversa con el usuario y va llenando la **ficha técnica** de su
   empresa (a la derecha se ve el avance ✅/⬜). Gemini solo *extrae* datos; **Python valida** rangos, tipos y departamentos
   (`core/perfil.py`) y decide qué falta y cuándo la ficha está completa. Si el modelo dice «confirmado» con datos
   faltantes, Python lo ignora.
2. **Evaluación (sin IA).** Con la ficha completa y confirmada (o con el botón **▶ Evaluar**):
   1. se obtienen los concursos **convocados** del OECE (descarga masiva o **API en vivo**);
   2. un **filtro determinístico** descarta lo que no es factible;
   3. un **sistema difuso de compatibilidad** (Mamdani, 0-100) puntúa cada concurso factible;
   4. se preseleccionan los mejores (umbral y Top-N) como **candidatos**;
   5. un **algoritmo genético** elige el **portafolio** (combinación de concursos) que conviene postular;
   6. dentro del fitness del AG, un **segundo sistema difuso** calcula el **riesgo del portafolio**, cuyo eje principal es
      la **cantidad de licitaciones** frente a la capacidad operativa de la empresa.
3. **Interpretación (Gemini).** El resultado, resumido en un JSON, vuelve al asistente, que lo explica en lenguaje sencillo
   y responde preguntas de seguimiento usando **solo** esos datos.

```
Fitness(x) = Σ compatibilidad(x) · (1 − Riesgo_difuso(x) / 100) ^ λ        (forma «descuento», por defecto)
Fitness(x) = Σ compatibilidad(x) − λ · Riesgo_difuso(x)                    (forma «resta», opcional)
```

> **Sin IA también funciona.** En la barra lateral puedes elegir una de las 3 empresas simuladas (TI, constructora,
> insumos médicos): el sistema corre igual y la explicación se muestra con el texto estándar (`core/explicacion.py`).
> Lo mismo ocurre si Gemini falla durante la interpretación: la app no se rompe y muestra esa explicación estándar.

Instalación detallada paso a paso (incluida la clave de Gemini) en [MANUAL_INSTALACION.md](MANUAL_INSTALACION.md).

---

## Requisitos

- Python **3.10 o superior** (probado con 3.12).
- Una **clave de API de Gemini** (gratuita, desde [Google AI Studio](https://aistudio.google.com/apikey)) **solo si vas a usar
  el asistente**. Para las empresas simuladas no hace falta.
- Internet: siempre para Gemini; para el OECE es opcional (hay una muestra de respaldo incluida).
- ~200 MB libres si descargas los datos completos del OECE (opcional).

## Instalación y ejecución

```bash
git clone https://github.com/Sergio-Osorio09/portafolio-concursos-oece.git
cd portafolio-concursos-oece

# (recomendado) entorno virtual
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt

cd app
streamlit run app.py
```

Se abre en **http://localhost:8501**.

> **Importante:** la app arranca en modo **💬 Asistente (Gemini)**. Si no hay clave configurada, verás un aviso y el chat
> estará deshabilitado: configura la clave (ver abajo) **o** elige una empresa simulada en la barra lateral para entrar
> directamente a los resultados.

### Clave de Gemini (elige una forma)

| Forma | Cómo |
|---|---|
| **Archivo de secretos** (recomendada) | Copia `app/.streamlit/secrets.toml.example` como `app/.streamlit/secrets.toml` y pega tu clave en `GEMINI_API_KEY` |
| Variable de entorno | `GEMINI_API_KEY` (también se acepta `GOOGLE_API_KEY`) |
| Desde la interfaz | Barra lateral → **🤖 Gemini → Clave de API** (solo dura mientras la sesión del navegador esté abierta) |

Prioridad: lo escrito en la barra lateral > `secrets.toml` > variable de entorno. La clave **nunca** va en el código;
`secrets.toml` y `.env` están en `.gitignore`.

**Modelo.** Por defecto se usa el alias `gemini-flash-latest` (siempre apunta al Flash vigente, porque `gemini-2.5-flash` tiene
anunciado su apagado para el 16-oct-2026). Si un modelo responde 404, la app prueba `gemini-3.5-flash` y luego
`gemini-2.5-flash`. Se puede forzar uno con `GEMINI_MODEL` (secretos o variable de entorno) o en la barra lateral.

**Cuota y privacidad.** Los límites del nivel gratuito los define Google. La app llama a Gemini solo cuando conversas y
**una vez** por resultado (no en cada movimiento de un control; usa **🔄 Reinterpretar** para regenerar la explicación).
Gemini recibe la ficha y el resumen del resultado (concursos elegidos, riesgo, sensibilidad), nunca la clave. Con el nivel
gratuito, Google puede usar lo enviado para mejorar sus productos: no escribas datos confidenciales de terceros.

### Datos de concursos

| Opción | Cómo | Cuándo usarla |
|---|---|---|
| **Muestra de respaldo** (incluida) | No hay que hacer nada: `app/data/muestra_respaldo.parquet` | Para probar de inmediato o sin internet |
| **Datos completos del OECE** | `python scripts/preparar_datos.py` (desde `app/`) | Descarga ~140 MB y los procesa (unos minutos). Genera el caché `app/data/oece_2026_procesos.parquet` |
| **API en vivo** | Barra lateral → *Fuente de concursos → API en vivo OECE* (slider de 5 a 60 páginas de 20 procesos; por defecto 25) | Procesos publicados hoy. Se cachea 15 min; si la API falla, la app vuelve sola a la descarga masiva |

La fuente por defecto es la **descarga masiva**: usa el caché completo si existe y, si no, la muestra de respaldo.
Aun con la API en vivo, la tabla de montos históricos (para estimar montos reservados) sale de la descarga masiva.

### Pruebas sin interfaz

Desde `app/`:

```bash
python scripts/prueba_flujo.py              # punta a punta con las 3 empresas simuladas
python scripts/prueba_asistente.py          # asistente SIMULADO: sin internet ni clave; debe terminar en «Todo OK ✔»
python scripts/prueba_asistente.py --real   # llama de verdad a Gemini (requiere GEMINI_API_KEY)
```

- `prueba_flujo.py` recorre filtro → compatibilidad → AG para cada empresa, compara el AG contra la búsqueda exhaustiva e
  imprime el barrido de λ (forma «resta» y «descuento»).
- `prueba_asistente.py` verifica la entrevista (extracción, validación, completitud), la confirmación prematura, el cambio
  al modelo de respaldo ante un 404, el manejo de errores (clave inválida, límite 429) y que el resultado del sistema se
  convierte en el JSON que recibe el asistente.

---

## Cómo usar la app

### Barra lateral

| Control | Qué hace |
|---|---|
| **Empresa** | «💬 Asistente (Gemini)» o una de las 3 empresas simuladas. Con el asistente aparece **↩️ Nueva conversación** |
| **🤖 Gemini** | Clave de API y modelo |
| **Fuente de concursos** | Descarga masiva OECE o API en vivo |
| **Riesgo en el fitness** | «Descuento» (`Σs·(1−R/100)^λ`) o «Resta» (`Σs − λ·R`) |
| **λ riesgo** | Aversión al riesgo (0-3 en descuento, 0-6 en resta; por defecto 1) |
| **Penalidades clásicas** | Opcional (apagado por defecto): recursos, fechas y ventana de disponibilidad. Castigan parte de lo mismo que el riesgo difuso (doble penalización) |
| **Parámetros avanzados** | Umbral de compatibilidad (60), Top-N candidatos (15), población (80), generaciones máx. (150) y probabilidad de mutación |

### Pestañas

Arriba hay una franja de resumen siempre visible (concursos vigentes, factibles, candidatos, licitaciones/capacidad,
riesgo y fitness).

| Pestaña | Qué hace |
|---|---|
| **💬 Asistente** | (solo en modo «Asistente») Explicación del resultado en lenguaje sencillo, botón **Reinterpretar** y chat de preguntas de seguimiento. Incluye la conversación inicial de recolección |
| **🗺️ Arquitectura** | Diagrama del flujo y fórmula del fitness |
| **🏢 Empresa** | Ficha técnica editable (en modo «Asistente» viene prellenada con lo que dijo el usuario) y la ficha estructurada en JSON |
| **📥 Concursos OECE** | Concursos factibles y embudo del filtro duro (cuántos descarta cada regla) |
| **🎯 Compatibilidad** | Ranking individual; al hacer clic en un concurso se ven sus entradas, las reglas activadas y el centroide. Funciones de pertenencia y reglas |
| **⚠️ Riesgo del portafolio** | Simulador del riesgo y curva **riesgo vs. cantidad de licitaciones** |
| **🧬 Algoritmo genético** | Convergencia, validación contra búsqueda exhaustiva y **barrido de λ** |
| **✅ Resultado** | Top 5 portafolios, descomposición del fitness, calendario del recomendado y explicación estándar |

Recorrido sugerido: elegir empresa (o conversar con el asistente) → **Riesgo del portafolio** (mover el nº de licitaciones por
encima de la capacidad) → **Algoritmo genético** (subir λ y ver cómo baja la cantidad de licitaciones) → **Resultado** →
**Asistente** (pedir que explique y preguntar).

---

## Cómo funciona por dentro

### 1. Ficha técnica y rol de la IA (`perfil.py`, `asistente.py`)
Campos obligatorios: rubro (palabras clave), categorías (bienes/servicios/obras), capacidad financiera, monto mínimo de
interés, presupuesto de postulaciones, años de experiencia, personal disponible, capacidad operativa, departamento base y
cobertura (nacional/regional; si es regional, también los departamentos).

- Gemini responde en JSON con el mensaje al usuario, las actualizaciones de campos y si el usuario confirmó el resumen.
- `perfil.limpiar_campos` normaliza lo que dijo el modelo: números («1.5 millones»), alias de categorías y departamentos,
  ajuste a los rangos permitidos (con aviso) y descarte de ceros inventados.
- `perfil.faltantes` y `perfil.validar` deciden (en Python) qué falta y si la ficha es consistente. La confirmación solo
  cuenta si la ficha está completa y válida.
- En la interpretación, el asistente recibe únicamente el JSON de resultados y tiene instrucción de no inventar cifras ni
  seguir instrucciones que vengan dentro de los textos de los concursos.

### 2. Filtro duro (`filtro.py`)
Descarta lo que no admite grados, en este orden (el embudo muestra cuántos quedan tras cada regla): estado inválido
(nulo/desierto/adjudicado…), categoría no atendida, contratación directa, monto no disponible, monto sobre la capacidad
financiera, monto menor al de interés, rubro ajeno (sin palabras clave) y, si la cobertura es regional, fuera de cobertura.
Los montos en USD se convierten a soles con un tipo de cambio fijo de 3.75.

### 3. Sistema difuso #1 — Compatibilidad (por concurso)
Mamdani con 6 entradas (ratio de monto, experiencia, plazo, afinidad técnica, personal, ubicación), 16 reglas, AND = mínimo,
agregación por máximo y defuzzificación por centroide. Salida: 0-100.

### 4. Preselección
Se toman los concursos con compatibilidad ≥ umbral (60) y se conservan los Top-N (15): son los **genes** del AG. Si menos de 3
superan el umbral, se toman los N mejores sin umbral (la app lo avisa).

### 5. Sistema difuso #2 — Riesgo del portafolio (por combinación)

| Entrada | Cálculo |
|---|---|
| `cantidad` | nº de licitaciones / capacidad operativa (0-2) |
| `capital` | monto total comprometido / capacidad financiera |
| `personal` | personal requerido / personal disponible |
| `fechas` | choque de cierres de postulación entre pares (dos cierres a menos de 7 días compiten por el mismo equipo de propuestas) |

Las reglas forman una **matriz por nivel de cantidad**: cada nivel (baja, media, alta) tiene un riesgo base y los demás
factores lo escalan; `cantidad EXCESIVA → riesgo CRÍTICO`, y capital o personal excedidos también son críticos. Se verificó
que en todo el espacio de entradas se activa al menos una regla. Niveles: bajo (<30), medio (<55), alto (<75), crítico.

### 6. Algoritmo genético (`portafolio_ag.py`)
- **Gen**: postular o no al concurso Cᵢ (0/1) · **Cromosoma**: portafolio de N concursos candidatos.
- Selección por torneo (3), cruce uniforme (p = 0.9), mutación bit a bit, elitismo (2), reparación y parada por paciencia
  (40 generaciones sin mejora). Semilla fija (42): los resultados son reproducibles.
- **Restricciones duras (vía reparación)**: entre 1 y 2× la capacidad operativa de licitaciones, y costo de postulación total
  ≤ presupuesto (costo por concurso = S/ 2 500 + 6 % del monto). Quién decide cuántas son «demasiadas» es la lógica difusa,
  no un corte fijo.
- **Validación**: si el espacio tiene ≤ 60 000 combinaciones, la app corre también la **búsqueda exhaustiva** y comprueba
  si el AG halló el óptimo global.
- **Barrido de λ**: corre el AG para varios valores de λ y muestra cuántas licitaciones elige y con qué riesgo.
- **Resta vs. descuento**: con la resta, Σs crece sin límite y el riesgo se satura en 100, así que el efecto de λ es todo o
  nada; el descuento produce una reducción gradual de la cantidad de licitaciones.

### 7. Tratamiento de los datos del OECE (`oece.py`)
- **Vigencia**: `tenderPeriod` trae inicio = fin, por eso se usa `items.statusDetails = CONVOCADO` y el cierre se estima como
  fin de consultas + 8 días (o publicación + 15 días si no hay etapa de consultas).
- **Montos reservados**: con la Ley 32069, la mayoría de los procesos de bienes y servicios convocados no publica el valor
  estimado. Se estima con la mediana de procesos ya adjudicados del mismo método y categoría, y se marca como estimado.
- **Referencia temporal**: con la descarga masiva (corte mensual) la fecha de referencia es la del corte, y si hay pocos
  vigentes se retrocede de 7 en 7 días; con la API en vivo la referencia es hoy.

---

## Estructura

```
portafolio-concursos-oece/
├── README.md
├── MANUAL_INSTALACION.md          # Instalación paso a paso, clave de Gemini y solución de problemas
├── GUION_PRESENTACION.md          # Guion para la exposición
├── requirements.txt
├── notebooks/                     # Notebooks originales del modelo
├── app/
│   ├── app.py                     # Interfaz Streamlit (entrevista, pestañas, barra lateral)
│   ├── .streamlit/
│   │   └── secrets.toml.example   # Plantilla de la clave de Gemini (copiar como secrets.toml)
│   ├── core/
│   │   ├── perfil.py              # Ficha técnica, empresas simuladas y validación de lo que extrae la IA
│   │   ├── asistente.py           # Cliente de Gemini: entrevista → ficha, e interpretación del resultado
│   │   ├── oece.py                # Adaptador OECE: descarga masiva, API en vivo, vigencia
│   │   ├── filtro.py              # Filtro determinístico y estimación de montos reservados
│   │   ├── difuso.py              # Motor Mamdani genérico
│   │   ├── compatibilidad.py      # Sistema difuso #1
│   │   ├── riesgo.py              # Sistema difuso #2 (riesgo del portafolio)
│   │   ├── portafolio_ag.py       # Algoritmo genético, búsqueda exhaustiva y barrido de λ
│   │   └── explicacion.py         # Explicación estándar del resultado (sin IA)
│   ├── scripts/
│   │   ├── preparar_datos.py      # Descarga y procesa los datos del OECE
│   │   ├── prueba_flujo.py        # Prueba de punta a punta sin interfaz
│   │   └── prueba_asistente.py    # Prueba del asistente (simulada; con --real llama a Gemini)
│   └── data/
│       └── muestra_respaldo.parquet
```

## Limitaciones

- Montos reservados, personal requerido, duración y fecha de cierre son **estimaciones** con supuestos explícitos.
- Las reglas y funciones de pertenencia se definieron a criterio y **no están calibradas** con datos. La siguiente etapa es
  calibrarlas con un AG offline sobre procesos históricos adjudicados.
- El asistente puede equivocarse al interpretar lo que dice el usuario: por eso la ficha se muestra en pantalla, es editable
  en la pestaña **Empresa** y pasa por validación en Python. La explicación de Gemini es una interpretación; **verifica siempre
  las bases de cada concurso en el SEACE antes de postular**.
- El asistente depende de un servicio externo (conexión, cuota y vigencia del modelo). Sin él, usa las empresas simuladas.

## Fuente de datos

Datos abiertos del **OECE** (Organismo Especializado para las Contrataciones Públicas Eficientes, Perú) en estándar OCDS,
publicados bajo licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/):
[contratacionesabiertas.oece.gob.pe](https://contratacionesabiertas.oece.gob.pe) y el
[Registro de Datos de Open Contracting](https://data.open-contracting.org/es/publication/135).
