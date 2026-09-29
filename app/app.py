"""Sistema Inteligente de Recomendación de Portafolios de Concursos (OECE)
Lógica difusa (compatibilidad) + Lógica difusa (riesgo del portafolio) + Algoritmo genético.

Ejecutar:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import compatibilidad as CO
from core import explicacion, filtro, oece, perfil
from core import portafolio_ag as PA
from core import riesgo as R
from core.difuso import regla_texto, trimf

st.set_page_config(page_title="Portafolio de Concursos OECE", page_icon="📊", layout="wide")

COLORES = ["#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#264653"]


# ============================ utilidades ============================
@st.cache_data(show_spinner="Cargando datos del OECE…")
def cargar_bulk():
    df, fuente = oece.cargar_bulk()
    return df, fuente, oece.tabla_montos_historicos(df)


@st.cache_data(ttl=900, show_spinner="Consultando la API en vivo del OECE…")
def cargar_api(paginas):
    return oece.consultar_api(paginas)


def grafico_mf(variables, universos, titulo_fn=lambda v: v, cols=3, altura=220, marcas=None):
    nombres = list(variables)
    filas = [nombres[i:i + cols] for i in range(0, len(nombres), cols)]
    for fila in filas:
        cs = st.columns(cols)
        for c, v in zip(cs, fila):
            lo, hi = universos[v]
            x = np.linspace(lo, hi, 300)
            fig = go.Figure()
            for i, (t, p) in enumerate(variables[v].items()):
                fig.add_trace(go.Scatter(x=x, y=trimf(x, *p), name=t, line=dict(color=COLORES[i % 5], width=2)))
            if marcas and v in marcas:
                fig.add_vline(x=marcas[v], line_dash="dash", line_color="black")
            fig.update_layout(title=titulo_fn(v), height=altura, margin=dict(l=10, r=10, t=35, b=10),
                              legend=dict(orientation="h", y=-0.25, font=dict(size=10)), yaxis_range=[0, 1.05])
            c.plotly_chart(fig, width="stretch")


def grafico_salida(sistema, res, titulo):
    fig = go.Figure()
    for i, (t, mf) in enumerate(sistema.mf_salida.items()):
        fig.add_trace(go.Scatter(x=sistema.Y, y=mf, name=t, line=dict(color=COLORES[i % 5], dash="dot")))
    fig.add_trace(go.Scatter(x=sistema.Y, y=res["agregada"], name="agregada", fill="tozeroy",
                             line=dict(color="#264653", width=3)))
    fig.add_vline(x=res["valor"], line_color="red", annotation_text=f"centroide = {res['valor']:.1f}")
    fig.update_layout(title=titulo, height=300, margin=dict(l=10, r=10, t=40, b=10), yaxis_range=[0, 1.05])
    return fig


def tabla_reglas(activas, salida):
    return pd.DataFrame([{"Regla": regla_texto(a, c, salida), "Activación (mín)": round(w, 3)} for a, c, w in activas])


# ============================ barra lateral ============================
st.sidebar.title("⚙️ Configuración")
fuente_sel = st.sidebar.radio("Fuente de concursos", ["Descarga masiva OECE (corte mensual)", "API en vivo OECE"],
                              help="Ambas son fuentes oficiales del OECE en formato OCDS.")
paginas = 25
if fuente_sel == "API en vivo OECE":
    paginas = st.sidebar.slider("Páginas a consultar (20 procesos c/u)", 5, 60, 25)

empresa_sel = st.sidebar.selectbox("Empresa simulada", list(perfil.EMPRESAS))
st.sidebar.caption("En el sistema completo, la ficha la construye el chatbot (IA generativa). "
                   "Aquí se simula con un formulario.")

try:
    df_bulk, fuente_bulk, montos_hist = cargar_bulk()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

if fuente_sel == "API en vivo OECE":
    try:
        df_src = cargar_api(paginas)
        fecha_ref = pd.Timestamp.today().normalize()
        vig, fecha_ref, corte, retro = oece.concursos_vigentes(df_src, fecha_ref=fecha_ref, min_vigentes=0, max_retro=0)
        fuente_txt = f"API en vivo ({len(df_src)} procesos recientes)"
    except Exception as e:
        st.sidebar.error(f"API no disponible ({e}). Se usa la descarga masiva.")
        fuente_sel = "Descarga masiva OECE (corte mensual)"
if fuente_sel != "API en vivo OECE":
    df_src = df_bulk
    vig, fecha_ref, corte, retro = oece.concursos_vigentes(df_src)
    fuente_txt = fuente_bulk
st.sidebar.success(f"**{fuente_txt}**\n\nFecha de referencia: {fecha_ref.date()}\n\nVigentes: {len(vig):,}")

# ============================ encabezado ============================
st.title("📊 Sistema Inteligente de Portafolio de Concursos Públicos")
st.caption("Lógica difusa (compatibilidad) → Lógica difusa (riesgo del portafolio) → Algoritmo genético · "
           "Datos abiertos oficiales del OECE (OCDS)")

tabs = st.tabs(["0 · Arquitectura", "1 · Empresa", "2 · Concursos OECE", "3 · Compatibilidad difusa",
                "4 · Riesgo difuso", "5 · Algoritmo genético", "6 · Resultado"])

# ============================ 0. Arquitectura ============================
with tabs[0]:
    if True:
        st.graphviz_chart("""
        digraph {
          rankdir=LR; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];
          F [label="1. Ficha técnica\\n(simula chatbot)", fillcolor="#e9f5f3"];
          O [label="2. Adaptador OECE\\nbulk + API en vivo", fillcolor="#e9f5f3"];
          D [label="Filtro duro\\n(restricciones)", fillcolor="#fff3d6"];
          C [label="3. Difuso #1\\nCompatibilidad\\n(por concurso)", fillcolor="#fde2d4"];
          P [label="Preselección\\nTop-N", fillcolor="#fff3d6"];
          G [label="5. Algoritmo genético\\n(elige el portafolio)", fillcolor="#dbe7f3"];
          RK [label="4. Difuso #2\\nRiesgo del portafolio\\n(cantidad, capital,\\npersonal, fechas)", fillcolor="#fde2d4"];
          S [label="6. Portafolio +\\nexplicación", fillcolor="#e9f5f3"];
          F -> D; O -> D; D -> C -> P -> G -> S;
          G -> RK [label="cada cromosoma", fontsize=9]; RK -> G [label="riesgo 0-100\\nen el fitness", fontsize=9];
        }""", width="stretch")
    if True:
        st.markdown("""
**Qué hace el sistema**

1. **Ficha técnica**: el perfil de la empresa (en el sistema completo la extrae el chatbot).
2. **OECE**: concursos reales convocados, desde la descarga masiva o la API en vivo.
3. **Difuso #1 (compatibilidad)**: puntúa **cada concurso** de 0 a 100 con 6 variables y 16 reglas Mamdani.
4. **Difuso #2 (riesgo)**: puntúa **cada combinación** de concursos de 0 a 100. El riesgo depende de la
   combinación: dos concursos seguros por separado pueden ser riesgosos juntos.
5. **Algoritmo genético**: busca la combinación con mejor **fitness**, que incluye el riesgo difuso.
6. **Resultado**: portafolio, alternativas y explicación.
""")
    st.info("**Lo pedido por el profesor — riesgo por cantidad de licitaciones dentro del fitness:**\n\n"
            "`Fitness(x) = Σ compatibilidad(x) · (1 − Riesgo_difuso(x)/100)^λ`  (o en forma de resta: `Σ compatibilidad − λ·Riesgo`)\n\n"
            "El **eje principal** del sistema de riesgo es `cantidad = nº licitaciones / capacidad operativa`. "
            "El tope duro del AG se amplió a 2× la capacidad, de modo que **es la lógica difusa, no una regla fija, "
            "la que decide cuántas licitaciones son demasiadas**.")

# ============================ 1. Empresa ============================
with tabs[1]:
    base = perfil.EMPRESAS[empresa_sel]
    k = empresa_sel  # las claves de los widgets dependen de la empresa: al cambiarla se recargan
    st.subheader(f"Ficha técnica — {empresa_sel}")
    c1, c2, c3 = st.columns(3)
    with c1:
        kws = st.text_area("Palabras clave del rubro (una por línea)", "\n".join(base["rubro_keywords"]), height=260, key=k + "kw")
        cats = st.multiselect("Categorías", list(perfil.CATEGORIAS), base["categorias"],
                              format_func=perfil.CATEGORIAS.get, key=k + "cat")
    with c2:
        cap_fin = st.number_input("Capacidad financiera (S/)", 10_000, 50_000_000, base["capacidad_financiera"], 50_000, key=k + "cf")
        m_min = st.number_input("Monto mínimo de interés (S/)", 0, 5_000_000, base["monto_minimo_interes"], 5_000, key=k + "mm")
        presup = st.number_input("Presupuesto para postulaciones (S/)", 5_000, 5_000_000, base["presupuesto_postulaciones"], 5_000, key=k + "pp")
        exp = st.number_input("Años de experiencia", 0, 60, base["experiencia_anios"], key=k + "ex")
    with c3:
        pers = st.number_input("Personal disponible", 1, 500, base["personal_disponible"], key=k + "pe")
        cap_op = st.number_input("Capacidad operativa (proyectos simultáneos)", 1, 20, base["capacidad_operativa"], key=k + "co",
                                 help="Es la referencia de la variable 'cantidad' del riesgo difuso.")
        dep = st.text_input("Departamento base", base["departamento_base"], key=k + "dep").upper().strip()
        cob = st.radio("Cobertura", ["nacional", "regional"], ["nacional", "regional"].index(base["cobertura"]), horizontal=True, key=k + "cob")
        deps = st.text_input("Departamentos de cobertura (coma)", ", ".join(base["departamentos_cobertura"]), key=k + "deps")

    ficha = {**base,
             "rubro_keywords": [w.strip() for w in kws.splitlines() if w.strip()],
             "categorias": cats, "capacidad_financiera": cap_fin, "monto_minimo_interes": m_min,
             "presupuesto_postulaciones": presup, "experiencia_anios": exp, "personal_disponible": pers,
             "capacidad_operativa": int(cap_op), "departamento_base": dep, "cobertura": cob,
             "departamentos_cobertura": [d.strip().upper() for d in deps.split(",") if d.strip()]}
    errores = perfil.validar(ficha)
    if errores:
        for e in errores:
            st.error(e)
        st.stop()
    with st.expander("Ficha estructurada (lo que el chatbot entregaría al núcleo inteligente)"):
        st.json(ficha)

# ============================ 2. Concursos ============================
fac, embudo = filtro.aplicar(vig, ficha, montos_hist)
ev = CO.evaluar(fac, ficha, fecha_ref) if len(fac) else fac

with tabs[2]:
    st.subheader("Concursos del OECE y filtro determinístico")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Procesos en la fuente", f"{len(df_src):,}")
    c2.metric("Vigentes (convocados)", f"{len(vig):,}")
    c3.metric("Factibles para la empresa", f"{len(fac):,}")
    c4.metric("Con monto estimado", f"{int(fac['monto_estimado'].sum()) if len(fac) else 0:,}")

    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure(go.Funnel(y=embudo["Etapa"], x=embudo["Quedan"], textinfo="value",
                                  marker=dict(color="#2a9d8f")))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), title="Embudo del filtro duro")
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("""
**¿Por qué un filtro duro antes de la lógica difusa?**
Hay condiciones que **no admiten grados**: si el concurso es de otro rubro, supera la capacidad
financiera o es contratación directa, no tiene sentido darle un puntaje parcial. El filtro los descarta
y la lógica difusa trabaja solo sobre lo factible.

**Dos hallazgos en los datos reales del OECE**
- `tenderPeriod` trae inicio = fin (solo la fecha de convocatoria). La **vigencia** se determina con
  `items.statusDetails = CONVOCADO` y el **cierre se estima** como fin de consultas + 8 días.
- Con la Ley 32069, el **valor estimado de bienes y servicios está reservado** mientras el proceso
  está convocado (~88 % viene en 0). En vez de descartarlos, se **estima el monto** con la mediana
  de procesos **ya adjudicados** del mismo método y categoría, y se marca como estimado.
""")
    if len(fac):
        st.dataframe(fac[["descripcion", "entidad", "departamento", "categoria", "metodo", "monto_pen",
                          "monto_estimado", "cierre_est", "afinidad"]].rename(columns={"monto_pen": "monto S/"}),
                     width="stretch", height=300)
    else:
        st.warning("Ningún concurso pasa el filtro. Amplíe palabras clave, categorías o capacidad financiera.")
        st.stop()

# ============================ 3. Compatibilidad ============================
with tabs[3]:
    st.subheader("Sistema difuso #1 — Compatibilidad empresa-concurso (0-100)")
    st.markdown("Mamdani: funciones triangulares → reglas SI-ENTONCES (AND = mínimo) → agregación por máximo → "
                "**centroide**. Se evalúa **cada concurso por separado**.")
    with st.expander("Variables de entrada y cómo se calculan", expanded=False):
        st.table(pd.DataFrame({"Variable": list(CO.DESCRIPCION), "Cálculo": list(CO.DESCRIPCION.values())}))
        grafico_mf(CO.VARIABLES, CO.UNIVERSOS)
        st.markdown("**Reglas**")
        st.table(pd.DataFrame({"Regla": [regla_texto(a, c, "compatibilidad") for a, c in CO.REGLAS]}))

    top = ev.head(50).copy()
    top["regla dominante"] = top["regla_dominante"].map(lambda r: regla_texto(r[0], r[1], "compat.") if r else "")
    st.markdown(f"**Ranking individual** ({len(ev)} concursos factibles; se muestran 50)")
    st.dataframe(top[["descripcion", "entidad", "departamento", "monto_pen", "score", "lectura"] + CO.ENTRADAS]
                 .rename(columns={"monto_pen": "monto S/"}), width="stretch", height=300)

    st.markdown("#### 🔍 Ver el razonamiento difuso de un concurso")
    idx = st.selectbox("Concurso", top.index, format_func=lambda i: f"{top.loc[i, 'score']:.0f} · {top.loc[i, 'descripcion'][:120]}")
    entrada = {v: float(ev.loc[idx, v]) for v in CO.ENTRADAS}
    res = CO.SISTEMA.evaluar(entrada, detalle=True)
    c1, c2 = st.columns([2, 3])
    with c1:
        st.dataframe(pd.DataFrame({"valor": entrada}).T.round(2), width="stretch")
        grados = pd.DataFrame(res["grados"]).T.round(2).fillna("")
        st.markdown("Grados de pertenencia")
        st.dataframe(grados, width="stretch")
    with c2:
        st.plotly_chart(grafico_salida(CO.SISTEMA, res, "Salida agregada y centroide"), width="stretch")
    st.dataframe(tabla_reglas(res["activas"], "compatibilidad"), width="stretch")

# ============================ 4. Riesgo ============================
with tabs[4]:
    st.subheader("Sistema difuso #2 — Riesgo del PORTAFOLIO (0-100)")
    st.markdown("Se evalúa una **combinación** de concursos. Su salida entra al **fitness** del algoritmo genético. "
                f"Con la capacidad operativa de esta empresa (**{ficha['capacidad_operativa']}**), "
                "`cantidad = nº licitaciones / capacidad`.")
    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("##### Simulador")
        kk = st.slider("Nº de licitaciones", 1, 2 * ficha["capacidad_operativa"], ficha["capacidad_operativa"])
        cap_r = st.slider("Capital comprometido / capacidad financiera", 0.0, 2.0, 0.3, 0.05)
        per_r = st.slider("Personal requerido / disponible", 0.0, 2.0, 0.4, 0.05)
        fec_r = st.slider("Choque de cierres de postulación", 0.0, 1.0, 0.1, 0.05)
        ent_r = {"cantidad": min(2.0, kk / ficha["capacidad_operativa"]), "capital": cap_r, "personal": per_r, "fechas": fec_r}
        rr = R.SISTEMA.evaluar(ent_r, detalle=True)
        st.metric("Riesgo difuso", f"{rr['valor']:.1f} / 100", R.nivel(rr["valor"]).upper(), delta_color="off")
    with c2:
        st.plotly_chart(grafico_salida(R.SISTEMA, rr, "Salida agregada del riesgo"), width="stretch")
    st.dataframe(tabla_reglas(rr["activas"], "riesgo"), width="stretch")

    st.markdown("##### 📈 Efecto de la CANTIDAD de licitaciones")
    st.caption("Cada licitación agrega una fracción de capital y personal; el choque de fechas se mantiene fijo. "
               "Pasada la capacidad operativa se activa CANTIDAD EXCESIVA → riesgo CRÍTICO.")
    fig = go.Figure()
    for nombre, (a, b, c) in {"licitaciones pequeñas": (0.05, 0.07, 0.05), "licitaciones medianas": (0.1, 0.15, 0.2),
                              "licitaciones grandes": (0.2, 0.25, 0.4)}.items():
        ks, vals = R.curva_por_cantidad(ficha["capacidad_operativa"], a, b, c)
        fig.add_trace(go.Scatter(x=ks, y=vals, mode="lines+markers", name=nombre))
    fig.add_vline(x=ficha["capacidad_operativa"], line_dash="dash", annotation_text="capacidad operativa")
    fig.update_layout(height=340, xaxis_title="nº de licitaciones en el portafolio", yaxis_title="riesgo difuso",
                      margin=dict(l=10, r=10, t=20, b=10), yaxis_range=[0, 100])
    st.plotly_chart(fig, width="stretch")

    with st.expander("Funciones de pertenencia y matriz de reglas del riesgo"):
        grafico_mf(R.VARIABLES, R.UNIVERSOS, cols=4, marcas=ent_r)
        st.table(pd.DataFrame({"Variable": list(R.DESCRIPCION), "Qué mide": list(R.DESCRIPCION.values())}))
        m = pd.DataFrame(R.MATRIZ).T
        m.columns = ["base", "+ capital ajustado", "+ personal justo", "+ fechas medio", "+ fechas alto"]
        m.loc["excesiva"] = ["critico"] + ["(máximo)"] * 4
        st.markdown("**Matriz de reglas** — cada nivel de cantidad tiene un riesgo base y los otros factores lo escalan. "
                    "Globales: capital excedido → crítico · personal excedido → crítico.")
        st.table(m)

# ============================ 5. Algoritmo genético ============================
with tabs[5]:
    st.subheader("Algoritmo genético — selección del portafolio con riesgo en el fitness")
    c1, c2, c3 = st.columns(3)
    with c1:
        umbral = st.slider("Umbral de compatibilidad para preselección", 0, 90, 60)
        top_n = st.slider("Top-N candidatos (longitud del cromosoma)", 5, 25, 15)
    with c2:
        forma = st.radio("Forma del riesgo en el fitness", ["descuento", "resta"], horizontal=True,
                         format_func={"descuento": "Descuento: Σs·(1−R/100)^λ", "resta": "Resta: Σs − λ·R"}.get)
        lam = st.slider("λ riesgo (aversión al riesgo)", 0.0, 6.0 if forma == "resta" else 3.0, 1.0, 0.25)
        pen = st.checkbox("Agregar penalidades clásicas (recursos, fechas, ventana)", False,
                          help="Ojo: castigan parte de lo mismo que el riesgo difuso (doble penalización).")
    with c3:
        pob = st.slider("Población", 20, 200, 80, 10)
        gens = st.slider("Generaciones máx.", 20, 400, 150, 10)
        pmut = st.slider("Prob. de mutación por gen", 0.01, 0.3, round(1 / top_n, 2), 0.01)

    cand = ev[ev["score"] >= umbral].head(top_n)
    if len(cand) < 3:
        st.warning(f"Menos de 3 concursos superan {umbral}; se toman los {top_n} mejores sin umbral.")
        cand = ev.head(top_n)
    cand = cand.reset_index(drop=True)
    cand.index = [f"C{i + 1}" for i in range(len(cand))]
    ctx = PA.construir_contexto(cand, ficha, fecha_ref)
    lambdas = {"riesgo": lam, "recursos": 1.0, "fechas": 0.3, "ventana": 0.2}
    fit = PA.Evaluador(ctx, lambdas, forma, pen)

    st.markdown(f"**Cromosoma:** {ctx['n']} genes binarios (1 = postular a Cᵢ) · restricción dura "
                f"1 ≤ Σx ≤ {ctx['k_max']} (2× capacidad) y costo de postulación ≤ S/ {ctx['presupuesto']:,.0f} · "
                f"espacio de búsqueda: **{PA.n_combinaciones(ctx):,} combinaciones**.")
    with st.expander("Candidatos (genes)"):
        st.dataframe(cand[["descripcion", "monto_pen", "monto_estimado", "score", "personal_req", "duracion_est", "cierre_est"]],
                     width="stretch")

    params = {"poblacion": pob, "generaciones": gens, "p_mut": pmut}
    res = PA.algoritmo_genetico(ctx, fit, params)
    st.session_state["resultado"] = (res, cand, ctx, fit)
    xb = res["ranking"][0][0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Licitaciones elegidas", int(xb.sum()))
    c2.metric("Riesgo difuso", f"{R.riesgo(xb, ctx):.1f}")
    c3.metric("Fitness", f"{res['ranking'][0][1]:.1f}")
    c4.metric("Generaciones / evaluaciones", f"{res['generaciones']} / {res['evaluaciones']:,}")

    h = res["hist"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=h["mejor"], name="mejor fitness"))
    fig.add_trace(go.Scatter(y=h["media"], name="fitness promedio"))
    fig.add_trace(go.Scatter(y=h["k_mejor"], name="nº licitaciones del mejor", yaxis="y2", line=dict(dash="dot")))
    fig.update_layout(height=320, title="Convergencia", xaxis_title="generación", margin=dict(l=10, r=10, t=40, b=10),
                      yaxis2=dict(overlaying="y", side="right", title="nº licitaciones", rangemode="tozero"))
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### ✅ Validación contra búsqueda exhaustiva")
        n_comb = PA.n_combinaciones(ctx)
        if n_comb <= 60_000:
            xe, fe, n_ev = PA.exhaustivo(ctx, PA.Evaluador(ctx, lambdas, forma, pen))
            ok = abs(fe - res["ranking"][0][1]) < 1e-6
            st.write(f"Exhaustivo: {n_ev:,} portafolios factibles evaluados → fitness óptimo **{fe:.2f}**")
            st.write(f"AG: **{res['evaluaciones']:,}** evaluaciones → fitness **{res['ranking'][0][1]:.2f}**")
            (st.success if ok else st.warning)("El AG encontró el óptimo global" if ok else "El AG quedó cerca del óptimo")
        else:
            st.info(f"{n_comb:,} combinaciones: la búsqueda exhaustiva ya no es práctica → aquí se justifica el AG.")
    with c2:
        st.markdown("##### 🎚️ Barrido de λ: ¿el riesgo reduce la cantidad de licitaciones?")
        valores = [0, 0.5, 1, 2, 3, 4, 6] if forma == "resta" else [0, 0.25, 0.5, 1, 1.5, 2, 3]
        bar = PA.barrido_lambda(ctx, lambdas, valores, forma, pen, params)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=bar["λ riesgo"].astype(str), y=bar["nº licitaciones"], name="nº licitaciones", marker_color="#2a9d8f"))
        fig.add_trace(go.Scatter(x=bar["λ riesgo"].astype(str), y=bar["riesgo difuso"], name="riesgo", yaxis="y2",
                                 mode="lines+markers", line=dict(color="#e76f51")))
        fig.update_layout(height=300, xaxis_title="λ", margin=dict(l=10, r=10, t=10, b=10),
                          yaxis=dict(title="nº licitaciones"), yaxis2=dict(overlaying="y", side="right", title="riesgo", range=[0, 100]))
        st.plotly_chart(fig, width="stretch")
    st.dataframe(bar, width="stretch", hide_index=True)
    st.caption("Con λ = 0 el AG ignora el riesgo y llena el portafolio hasta el tope. Al subir λ el riesgo por cantidad "
               "pesa más y el AG elige menos licitaciones. **Resta vs. descuento:** en la resta Σs crece sin límite pero "
               "el riesgo se satura en 100, por eso el cambio es brusco (todo o nada); el descuento se interpreta como "
               "valor esperado y el ajuste es más gradual.")

# ============================ 6. Resultado ============================
with tabs[6]:
    res, cand, ctx, fit = st.session_state["resultado"]
    st.subheader("Portafolio recomendado y alternativas")
    filas = []
    for i, (x, fx) in enumerate(res["ranking"][:5], 1):
        d = fit.descomposicion(x)
        filas.append({"opción": f"Opción {i}", "concursos": " + ".join(cand.index[x == 1]), "nº": int(x.sum()),
                      **{k2: round(v, 1) for k2, v in d.items()}, "riesgo": round(R.riesgo(x, ctx), 1),
                      "fitness": round(fx, 1)})
    alt = pd.DataFrame(filas)
    st.dataframe(alt, width="stretch", hide_index=True)

    comp = alt.melt(id_vars="opción", value_vars=[c for c in alt.columns if c.startswith("Σ") or c.startswith("−")],
                    var_name="término", value_name="aporte")
    fig = px.bar(comp, x="opción", y="aporte", color="término", barmode="relative",
                 color_discrete_sequence=COLORES, title="Descomposición del fitness (verde suma, resto resta)")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")

    xb = res["ranking"][0][0]
    sel = cand[xb == 1]
    g = pd.DataFrame([{"concurso": cid, "fase": "preparación", "inicio": fecha_ref, "fin": ctx["cierre"][i]}
                      for i, cid in enumerate(cand.index) if xb[i]] +
                     [{"concurso": cid, "fase": "ejecución", "inicio": ctx["ejec_ini"][i], "fin": ctx["ejec_fin"][i]}
                      for i, cid in enumerate(cand.index) if xb[i]])
    fig = px.timeline(g, x_start="inicio", x_end="fin", y="concurso", color="fase", title="Calendario estimado del portafolio",
                      color_discrete_sequence=["#e9c46a", "#2a9d8f"])
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")

    st.markdown(explicacion.texto(xb, cand, ctx, fit, ficha))
