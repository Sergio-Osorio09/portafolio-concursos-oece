# Guion de presentación — Portafolio de Concursos OECE
**Lógica difusa (compatibilidad) + Lógica difusa (riesgo del portafolio) + Algoritmo genético**

> Cómo abrirlo: `cd app` → `streamlit run app.py` → http://localhost:8501
> Para el asistente: configurar la clave de Gemini antes de la clase (ver `MANUAL_INSTALACION.md`).
> (Si no hay internet: la app usa el caché local `data/oece_2026_procesos.parquet`, o la muestra de respaldo.)
>
> **Distribución de la pantalla:** arriba hay una **franja de resumen siempre visible** (vigentes, factibles, candidatos,
> licitaciones elegidas / capacidad, riesgo, fitness). En la **barra lateral** están la empresa, la fuente de datos y los
> controles del modelo (**λ**, forma del fitness, penalidades y parámetros avanzados del AG). Al mover λ, la franja
> de resumen se actualiza sin cambiar de pestaña.

---

## Apertura · Asistente Gemini — entrevista (2 min)
**Qué mostrar:** en la barra lateral, **Empresa → 💬 Asistente (Gemini)**. Responder 3 o 4 mensajes describiendo una
empresa (por ejemplo, una de TI en Lima con 1,5 millones de capacidad financiera y 3 proyectos a la vez). A la derecha
se ve la ficha llenándose (✅ / ⬜). Al terminar, confirmar ("sí, evalúa") o pulsar **▶ Evaluar**.

**Qué decir:**
- Es la capa de **IA generativa** del documento de arquitectura: convierte lenguaje natural en una ficha estructurada.
- **La IA no decide nada**: solo extrae datos. **Python valida** rangos, números ("1,5 millones" → 1 500 000) y
  departamentos, y es Python quien decide qué falta y cuándo la ficha está completa. Si el modelo dice "confirmado"
  pero faltan datos, se ignora.
- La clave de la API no está en el código (barra lateral, `secrets.toml` o variable de entorno).

**Plan B:** si no hay internet o se agota la cuota gratuita, elegir una **empresa simulada** en la barra lateral: el
núcleo inteligente funciona igual, sin IA.

---

## 0 · Arquitectura (1 min)
**Qué mostrar:** el diagrama y el recuadro azul.

**Qué decir:**
- El sistema tiene **dos sistemas difusos** y un **algoritmo genético**:
  1. Difuso #1 puntúa **cada concurso** (qué tan compatible es con la empresa).
  2. El AG busca **la mejor combinación** de concursos (el portafolio).
  3. Difuso #2 puntúa el **riesgo de cada combinación** que propone el AG, y ese riesgo **entra al fitness**.
- **Lo pedido por el profesor:** el riesgo por **cantidad de licitaciones** ahora está dentro del fitness:
  `Fitness = Σ compatibilidad · (1 − Riesgo/100)^λ`, y el eje principal del riesgo es
  `cantidad = nº licitaciones / capacidad operativa`.
- La IA generativa (asistente Gemini) entrevista al usuario para armar la ficha y al final explica el resultado.
  **La IA nunca calcula puntajes** (decisión 17 del documento).

**Por qué dos sistemas difusos y no uno:** la compatibilidad es una propiedad de *un* concurso; el riesgo es
una propiedad de *la combinación*. Dos concursos que solos son seguros pueden ser riesgosos juntos (suman
capital, personal y se cruzan en fechas). No se puede calcular el riesgo concurso por concurso.

---

## 1 · Empresa (1 min)
**Qué mostrar:** elegir en la barra lateral las 3 empresas simuladas (TI, constructora, insumos médicos).
**Qué decir:** estos campos son exactamente lo que el chatbot entregaría ("ficha estructurada" al final).
El campo clave para hoy es **capacidad operativa** (proyectos simultáneos): es la referencia de "cantidad".

---

## 2 · Concursos OECE (2 min)
**Qué mostrar:** métricas, embudo, y la fuente en la barra lateral (se puede cambiar a **API en vivo**).

**Qué decir:**
- **Dos fuentes oficiales**, mismo formato OCDS: la descarga masiva del OECE (corte mensual, ~122 000 procesos)
  y la **API en vivo** `contratacionesabiertas.oece.gob.pe/api/v1/releases`. Un **adaptador** aísla la API del núcleo.
- **Filtro duro antes de la lógica difusa:** lo que *no admite grados* (otro rubro, supera la capacidad
  financiera, contratación directa, fuera de cobertura) se descarta. El embudo muestra cuántos elimina cada regla.
- **Dos hallazgos con los datos reales** (muestran que trabajamos con los datos de verdad):
  1. `tenderPeriod` viene con inicio = fin. La **vigencia** se toma de `items.statusDetails = CONVOCADO`
     y el **cierre se estima** como fin de consultas + 8 días.
  2. Con la **Ley 32069** el valor estimado de bienes y servicios está **reservado** mientras está convocado
     (~88 % en 0). En lugar de descartarlos, **estimamos el monto** con la mediana de procesos ya adjudicados
     del mismo método y categoría, y lo marcamos como estimado.

---

## 3 · Compatibilidad difusa (2 min)
**Qué mostrar:** hacer **clic en una fila** del ranking: a la derecha aparecen sus entradas, la salida agregada con
el centroide y las reglas activadas. Abajo, el desplegable muestra las funciones de pertenencia y las 16 reglas.

**Qué decir:**
- 6 entradas: ratio de monto, experiencia, plazo, afinidad técnica, personal y ubicación → 16 reglas Mamdani.
- Proceso Mamdani: **fuzzificación** (grados de pertenencia) → **reglas** con AND = mínimo → **agregación**
  por máximo → **centroide** = puntaje 0-100.
- En el gráfico se ve el área agregada y la línea roja del centroide; en la tabla, qué reglas se activaron y con
  qué fuerza. **Es explicable**: se puede decir por qué un concurso tiene 73 y otro 55.

**Por qué lógica difusa:** "monto adecuado" o "experiencia suficiente" no tienen un corte exacto. Un concurso
con 4,9 años de experiencia requerida no debería pasar de 100 a 0 frente a uno de 5,1.

---

## 4 · Riesgo difuso — LO PEDIDO POR EL PROFESOR (3 min)
**Qué mostrar:**
1. Mover el slider **Nº de licitaciones** por encima de la capacidad: el riesgo salta a crítico y aparece la regla
   `SI cantidad es EXCESIVA ENTONCES riesgo es CRÍTICO`.
2. El gráfico **Efecto de la CANTIDAD**: el riesgo sube con cada licitación y se dispara al pasar la línea de
   capacidad operativa.
3. Abrir "Funciones de pertenencia y **matriz de reglas**".

**Qué decir:**
- 4 entradas del portafolio: **cantidad** (nº / capacidad operativa), capital comprometido, personal requerido y
  choque de cierres de postulación.
- **Cambios respecto a la semana pasada:**
  - Antes `cantidad` se cortaba en 1 y el AG no podía pasar de K por una regla fija, **así que el riesgo por
    cantidad nunca actuaba**. Ahora `cantidad` va de 0 a 2, hay un término nuevo **EXCESIVA** y el tope duro
    se amplió a 2× la capacidad: **es la lógica difusa la que decide cuántas son demasiadas**.
  - Las reglas se organizaron como una **matriz por nivel de cantidad**: cada nivel tiene un riesgo base
    (baja → bajo, media → bajo, alta → medio, excesiva → crítico) y capital, personal y fechas lo **escalan**
    un nivel. Así la cantidad es el eje principal y **no quedan huecos** (lo verificamos con una grilla de
    ~50 000 combinaciones de entradas: en todas se activa al menos una regla).
  - `fechas` ahora mide el **choque de cierres** (tener que preparar varias propuestas la misma semana).
    Con los datos reales, todos los concursos vigentes se ejecutan en el mismo periodo (solape ≈ 1 para
    cualquier par), y esa simultaneidad **ya la mide la cantidad frente a la capacidad operativa**: medirla
    dos veces duplicaba la penalización.

---

## 5 · Algoritmo genético (3 min)
**Qué mostrar:** métricas, convergencia, validación exhaustiva y el **barrido de λ**. En la barra lateral,
cambiar la forma a "Resta" para comparar, y mover λ mirando la franja de resumen.

**Qué decir:**
- **Gen** = postular o no a Cᵢ (0/1). **Cromosoma** = cadena binaria de N genes = un portafolio.
  **Población** = 80 portafolios.
- **Operadores:** selección por torneo (3), cruce uniforme, mutación bit a bit (1/N), elitismo (2),
  **reparación** (respeta el presupuesto de postulación y el tope de 2× la capacidad), parada por paciencia.
- **Fitness:** `Σ compatibilidad · (1 − Riesgo_difuso/100)^λ`. λ = **aversión al riesgo** de la empresa.
- **Validación:** con N = 15 hay ~10 000 combinaciones. La búsqueda exhaustiva las evalúa todas y el AG encuentra
  **el mismo óptimo** evaluando solo una fracción. Con N = 25 el espacio crece a ~240 000 y ahí se justifica el AG.
- **Barrido de λ (el gráfico clave):** con λ = 0 el AG ignora el riesgo y llena el portafolio hasta el tope.
  Al subir λ **elige menos licitaciones y el riesgo baja**. Esto demuestra que el riesgo por cantidad actúa
  dentro del fitness.

**Por qué "descuento" y no "resta" (decisión de diseño que se puede mostrar en vivo):**
con la resta (`Σs − λ·R`), Σs crece sin límite con cada licitación, pero R se satura en 100. Resultado: λ produce
un salto todo o nada (6 licitaciones → 2). Con el descuento, cada licitación extra solo conviene si su aporte supera lo
que el riesgo le quita a **todo** el portafolio. Se interpreta como **valor esperado** (1 − R/100 ≈ probabilidad
de ejecutar bien) y la reducción es gradual (6 → 4 → 2).

**Casilla "penalidades clásicas":** el notebook anterior restaba además penalidades de recursos y fechas.
Se dejó como opción, pero por defecto está apagada porque castiga dos veces lo mismo que el riesgo difuso.

---

## 6 · Resultado (1 min)
**Qué mostrar:** las 5 alternativas, la **descomposición del fitness** (cuánto suma la compatibilidad y cuánto
resta el riesgo), el calendario y la explicación en texto.
**Qué decir:** el usuario no recibe una sola respuesta sino alternativas justificadas, con la regla de riesgo
dominante y los supuestos declarados.

---

## Cierre · Asistente Gemini — interpretación (1-2 min)
**Qué mostrar:** pestaña **💬 Asistente**. Gemini explica el resultado en lenguaje sencillo. Hacer una pregunta de
seguimiento, por ejemplo "¿por qué no me recomiendas 3 concursos?". Si se cambia λ, aparece el aviso de que la
explicación quedó desactualizada y el botón **🔄 Reinterpretar**.

**Qué decir:**
- El asistente recibe **solo un JSON** con lo que calculó el sistema (portafolio, riesgo, alternativas, barrido de λ,
  embudo del filtro) y tiene la instrucción de no inventar cifras ni concursos.
- Así queda la separación del documento: **IA generativa = interacción y explicación; lógica difusa = evaluación;
  AG = optimización**.
- Si Gemini falla, se muestra la explicación estándar del sistema (plantilla), sin romper la app.

---

## Limitaciones (para decirlo antes de que pregunten)
- Montos reservados, personal requerido, duración y fecha de cierre son **estimaciones** con supuestos explícitos.
- Las reglas y funciones de pertenencia son **expertas, no calibradas**. El documento de arquitectura propone
  calibrarlas con un AG offline (fitness de ranking del ganador sobre datos históricos con `awards`): es la
  siguiente etapa.
- El asistente Gemini depende de internet y de la cuota gratuita de la API; las empresas simuladas son el respaldo.

## Preguntas probables
- **¿Por qué AG y no fuerza bruta?** El espacio crece como 2ᴺ; con N moderado ya no se puede enumerar. Aquí lo
  validamos contra la fuerza bruta en un tamaño donde todavía se puede.
- **¿Por qué el riesgo como factor del fitness y no como restricción?** Una restricción es un corte binario
  ("máximo 3"). El riesgo difuso permite que una 4.ª licitación muy compatible y barata entre si lo vale, y que
  una 3.ª muy pesada quede fuera.
- **¿Cómo se elige λ?** Es una preferencia de la empresa (aversión al riesgo). El barrido muestra el efecto de
  cada valor para que la empresa decida.
