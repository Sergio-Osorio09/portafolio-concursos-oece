# Sistema Inteligente de Portafolio de Concursos Públicos (OECE)

**Lógica difusa + Algoritmo genético con riesgo difuso en el fitness**, aplicado a los datos abiertos oficiales
de contrataciones públicas del Perú (OECE / SEACE, formato OCDS).

Dada la ficha técnica de una empresa (simulada), el sistema:

1. obtiene los concursos **convocados** desde el OECE (descarga masiva o **API en vivo**);
2. descarta los que no son factibles con un **filtro determinístico**;
3. puntúa cada concurso con un **sistema difuso de compatibilidad** (Mamdani, 0-100);
4. usa un **algoritmo genético** para elegir el **portafolio** (combinación de concursos) que conviene postular;
5. dentro del fitness del AG, un **segundo sistema difuso** calcula el **riesgo del portafolio**, cuyo eje
   principal es la **cantidad de licitaciones** frente a la capacidad operativa de la empresa.

```
Fitness(x) = Σ compatibilidad(x) · (1 − Riesgo_difuso(x) / 100) ^ λ
```

> Proyecto del curso **Software Inteligente**. El módulo de IA generativa (chatbot) queda fuera del alcance:
> la ficha técnica se ingresa con un formulario que simula la salida del chatbot.

---

## Requisitos

- Python **3.10 o superior** (probado con 3.12)
- ~200 MB libres si se descargan los datos completos del OECE (opcional)

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

### Datos

| Opción | Cómo | Cuándo usarla |
|---|---|---|
| **Muestra de respaldo** (incluida) | No hay que hacer nada: `app/data/muestra_respaldo.parquet` | Para probar de inmediato o sin internet |
| **Datos completos del OECE** | `python scripts/preparar_datos.py` (desde `app/`) | Descarga ~140 MB y los procesa en ~20 s. Genera el caché `app/data/oece_2026_procesos.parquet` |
| **API en vivo** | En la barra lateral: *Fuente de concursos → API en vivo OECE* | Consulta los procesos más recientes publicados hoy |

La app usa automáticamente el caché completo si existe; si no, usa la muestra de respaldo.

### Prueba sin interfaz

```bash
cd app
python scripts/prueba_flujo.py
```

Recorre todo el flujo con las 3 empresas simuladas, compara el AG contra la búsqueda exhaustiva e imprime
el barrido de λ.

---

## Cómo usar la app

| Pestaña | Qué hace |
|---|---|
| **0 · Arquitectura** | Diagrama del flujo y fórmula del fitness |
| **1 · Empresa** | Ficha técnica editable. Hay 3 empresas simuladas en la barra lateral (TI, constructora, insumos médicos) |
| **2 · Concursos OECE** | Concursos vigentes y embudo del filtro duro (cuántos descarta cada regla) |
| **3 · Compatibilidad difusa** | Funciones de pertenencia, reglas y ranking. Al elegir un concurso se ven sus grados, las reglas activadas y el centroide |
| **4 · Riesgo difuso** | Simulador del riesgo del portafolio y curva **riesgo vs. cantidad de licitaciones** |
| **5 · Algoritmo genético** | Parámetros, convergencia, validación contra búsqueda exhaustiva y **barrido de λ** |
| **6 · Resultado** | Top 5 portafolios, descomposición del fitness, calendario y explicación en texto |

Recorrido sugerido: elegir una empresa → pestaña 4 (mover el nº de licitaciones por encima de la capacidad) →
pestaña 5 (subir λ y ver cómo baja la cantidad de licitaciones elegidas) → pestaña 6.

---

## Cómo funciona

### Sistema difuso #1 — Compatibilidad (por concurso)
Mamdani con 6 entradas (ratio de monto, experiencia, plazo, afinidad técnica, personal, ubicación), 16 reglas,
AND = mínimo, agregación por máximo y defuzzificación por centroide. Salida: 0-100.

### Sistema difuso #2 — Riesgo del portafolio (por combinación)
| Entrada | Cálculo |
|---|---|
| `cantidad` | nº de licitaciones / capacidad operativa (0-2) |
| `capital` | monto total comprometido / capacidad financiera |
| `personal` | personal requerido / personal disponible |
| `fechas` | choque de cierres de postulación entre pares |

Las reglas forman una **matriz por nivel de cantidad**: cada nivel (baja, media, alta) tiene un riesgo base y
los demás factores lo escalan; `cantidad EXCESIVA → riesgo CRÍTICO`. Se verificó que en todo el espacio de
entradas se activa al menos una regla.

### Algoritmo genético
- **Gen**: postular o no al concurso Cᵢ (0/1) · **Cromosoma**: portafolio de N concursos candidatos.
- Selección por torneo, cruce uniforme, mutación bit a bit, elitismo, reparación (presupuesto de postulación y
  tope de 2× la capacidad) y parada por paciencia.
- El riesgo entra al fitness como **descuento** (valor esperado) o como **resta** (`Σs − λ·R`), a elección.
  Con la resta, Σs crece sin límite y el riesgo se satura en 100, así que el efecto de λ es todo o nada;
  el descuento produce una reducción gradual de la cantidad de licitaciones.

### Tratamiento de los datos del OECE
- **Vigencia**: `tenderPeriod` trae inicio = fin, por eso se usa `items.statusDetails = CONVOCADO` y el cierre se
  estima como fin de consultas + 8 días.
- **Montos reservados**: con la Ley 32069, la mayoría de los procesos de bienes y servicios convocados no publica
  el valor estimado. Se estima con la mediana de procesos ya adjudicados del mismo método y categoría, y se marca
  como estimado.

---

## Estructura

```
app/
├── app.py                    # Interfaz Streamlit
├── core/
│   ├── perfil.py             # Ficha técnica y empresas simuladas (Profile Service)
│   ├── oece.py               # Adaptador OECE: descarga masiva, API en vivo, vigencia
│   ├── filtro.py             # Filtro determinístico y estimación de montos reservados
│   ├── difuso.py             # Motor Mamdani genérico
│   ├── compatibilidad.py     # Sistema difuso #1
│   ├── riesgo.py             # Sistema difuso #2 (riesgo del portafolio)
│   ├── portafolio_ag.py      # Algoritmo genético, búsqueda exhaustiva y barrido de λ
│   └── explicacion.py        # Texto explicativo del resultado
├── scripts/
│   ├── preparar_datos.py     # Descarga y procesa los datos del OECE
│   └── prueba_flujo.py       # Prueba de punta a punta sin interfaz
└── data/
    └── muestra_respaldo.parquet
notebooks/                    # Notebooks originales del modelo
GUION_PRESENTACION.md         # Guion para la exposición
```

## Limitaciones
- Montos reservados, personal requerido, duración y fecha de cierre son **estimaciones** con supuestos explícitos.
- Las reglas y funciones de pertenencia se definieron a criterio y **no están calibradas** con datos. La siguiente
  etapa es calibrarlas con un AG offline sobre procesos históricos adjudicados.

## Fuente de datos
Datos abiertos del **OECE** (Organismo Especializado para las Contrataciones Públicas Eficientes, Perú) en
estándar OCDS, publicados bajo licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/):
[contratacionesabiertas.oece.gob.pe](https://contratacionesabiertas.oece.gob.pe) y el
[Registro de Datos de Open Contracting](https://data.open-contracting.org/es/publication/135).
