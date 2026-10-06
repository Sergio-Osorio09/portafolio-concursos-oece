"""Sistema Inteligente de Recomendación de Portafolios de Concursos (OECE)
Lógica difusa (compatibilidad) + Lógica difusa (riesgo del portafolio) + Algoritmo genético.

Ejecutar:  streamlit run app.py
"""
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import asistente, explicacion, filtro, oece, perfil
from core import compatibilidad as CO
from core import portafolio_ag as PA
from core import riesgo as R
from core.difuso import regla_texto, trimf

st.set_page_config(page_title="Portafolio de Concursos OECE", page_icon="📊", layout="wide")

COLORES = ["#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#8ab4f8"]
LINEA = "#9aa0a6"   # visible en tema claro y oscuro

st.markdown("""
<style>
.block-container {padding-top: 2.6rem; padding-bottom: 1rem; max-width: 100%;}
h1 {font-size: 1.55rem !important; padding: 0 !important; margin: 0 !important;}
h3 {font-size: 1.1rem !important; padding: .2rem 0 !important;}
h5 {font-size: .95rem !important; padding: .1rem 0 !important; margin: 0 !important;}
[data-testid="stMetricValue"] {font-size: 1.3rem;}
[data-testid="stMetricLabel"] p {font-size: .76rem;}
[data-testid="stMetric"] {padding: .15rem .6rem; border-left: 3px solid #2a9d8f; background: rgba(127,127,127,.07); border-radius: 4px;}
.stTabs [data-baseweb="tab-list"] {gap: .3rem;}
.stTabs [data-baseweb="tab"] {padding: .35rem .8rem; height: auto;}
div[data-testid="stExpander"] details summary p {font-size: .9rem;}
section[data-testid="stSidebar"] .block-container {padding-top: 1rem;}
</style>""", unsafe_allow_html=True)


# ============================ utilidades ============================
@st.cache_data(show_spinner="Cargando datos del OECE…")
def cargar_bulk():
    df, fuente = oece.cargar_bulk()
    return df, fuente, oece.tabla_montos_historicos(df)


@st.cache_data(ttl=900, show_spinner="Consultando la API en vivo del OECE…")
def cargar_api(paginas):
    return oece.consultar_api(paginas)


@st.cache_data(show_spinner="Filtrando y evaluando compatibilidad difusa…")
def evaluar_concursos(vig, ficha, montos_hist, fecha_ref):
    fac, embudo = filtro.aplicar(vig, ficha, montos_hist)
    ev = CO.evaluar(fac, ficha, fecha_ref) if len(fac) else fac
    return fac, embudo, ev


@st.cache_data(show_spinner="Ejecutando el algoritmo genético…")
def correr_modelo(cand, ficha, fecha_ref, lambdas, forma, pen, params):
    ctx = PA.construir_contexto(cand, ficha, fecha_ref)
    res = PA.algoritmo_genetico(ctx, PA.Evaluador(ctx, lambdas, forma, pen), params)
    exh = PA.exhaustivo(ctx, PA.Evaluador(ctx, lambdas, forma, pen)) if PA.n_combinaciones(ctx) <= 60_000 else None
    return ctx, res, exh


@st.cache_data(show_spinner="Barrido de λ…")
def correr_barrido(cand, ficha, fecha_ref, forma, pen, params):
    """No depende de λ: no se recalcula al mover el slider de λ."""
    ctx = PA.construir_contexto(cand, ficha, fecha_ref)
    valores = [0, 0.5, 1, 2, 3, 4, 6] if forma == "resta" else [0, 0.25, 0.5, 1, 1.5, 2, 3]
    return PA.barrido_lambda(ctx, {"recursos": 1.0, "fechas": 0.3, "ventana": 0.2}, valores, forma, pen, params)


def estilo(fig, alto, titulo=None, leyenda=True):
    fig.update_layout(height=alto, margin=dict(l=0, r=0, t=28 if titulo else 6, b=0),
                      title=dict(text=titulo or "", font=dict(size=13), x=0, y=.98, yanchor="top"),
                      font=dict(size=11), showlegend=leyenda,
                      legend=dict(orientation="h", yanchor="top", y=-0.24, xanchor="left", x=0, font=dict(size=10), title_text=""))
    fig.update_xaxes(title_standoff=4)
    return fig


def fig_mf(variables, universo, v, marca=None, alto=150):
    lo, hi = universo
    x = np.linspace(lo, hi, 250)
    fig = go.Figure()
    for i, (t, p) in enumerate(variables.items()):
        y = trimf(x, *p)
        fig.add_trace(go.Scatter(x=x, y=y, name=t, line=dict(color=COLORES[i % 5], width=2)))
        fig.add_annotation(x=x[int(np.argmax(y))], y=1.1, text=t, showarrow=False, font=dict(size=9, color=COLORES[i % 5]))
    if marca is not None:
        fig.add_vline(x=marca, line_dash="dash", line_color=LINEA)
    fig.update_yaxes(range=[0, 1.2], showticklabels=False)
    return estilo(fig, alto, v, leyenda=False)


def grilla_mf(variables, universos, cols, marcas=None):
    nombres = list(variables)
    for i in range(0, len(nombres), cols):
        for c, v in zip(st.columns(cols), nombres[i:i + cols]):
            c.plotly_chart(fig_mf(variables[v], universos[v], v, (marcas or {}).get(v)), width="stretch")


def fig_salida(sistema, res, alto=230, titulo=None):
    fig = go.Figure()
    for i, (t, mf) in enumerate(sistema.mf_salida.items()):
        fig.add_trace(go.Scatter(x=sistema.Y, y=mf, name=t, showlegend=False,
                                 line=dict(color=COLORES[i % 5], dash="dot", width=1)))
        fig.add_annotation(x=sistema.Y[int(np.argmax(mf))], y=1.06, text=t, showarrow=False,
                           font=dict(size=9, color=COLORES[i % 5]))
    fig.add_trace(go.Scatter(x=sistema.Y, y=res["agregada"], name="agregada", fill="tozeroy",
                             line=dict(color="#8ab4f8", width=2.5)))
    fig.add_vline(x=res["valor"], line_color="#e76f51", line_width=2,
                  annotation_text=f"centroide {res['valor']:.1f}", annotation_position="top left")
    fig.update_yaxes(range=[0, 1.12])
    return estilo(fig, alto, titulo, leyenda=False)


def tabla_reglas(activas, salida):
    return pd.DataFrame([{"Regla": regla_texto(a, c, salida), "Activación": round(w, 2)} for a, c, w in activas])


COL_BARRA = lambda t: st.column_config.ProgressColumn(t, min_value=0, max_value=100, format="%.0f")  # noqa: E731
COL_MONTO = st.column_config.NumberColumn("monto S/", format="compact")

# ============================ asistente (Gemini) ============================
MODO_ASISTENTE = "💬 Asistente IA"
SALUDO = ("¡Hola! Soy el asistente del sistema de portafolios de concursos públicos. Te haré unas preguntas sobre "
          "tu empresa para evaluar a qué concursos del OECE conviene postular. Para empezar: "
          "**¿a qué se dedica tu empresa y qué bienes, servicios u obras suele ofrecer?**")


def _secreto(nombre):
    try:
        return str(st.secrets.get(nombre, "") or "")
    except Exception:          # no existe .streamlit/secrets.toml
        return ""


def clave_gemini():
    """Clave de la API: lo escrito en la barra lateral > secrets.toml > variable de entorno. Nunca va en el código."""
    return (st.session_state.get("gemini_key_input", "").strip()
            or _secreto("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
            or _secreto("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", ""))


def _modelo_inicial():
    if asistente.es_openrouter(clave_gemini()):
        return _secreto("OPENROUTER_MODEL") or os.environ.get("OPENROUTER_MODEL", "") or asistente.OPENROUTER_DEFECTO
    return _secreto("GEMINI_MODEL") or os.environ.get("GEMINI_MODEL", "") or asistente.MODELO_DEFECTO


def modelo_gemini():
    return st.session_state.get(f"gemini_modelo_{asistente.proveedor(clave_gemini())}", "").strip() or _modelo_inicial()


def iniciar_chat():
    ss = st.session_state
    ss.chat_msgs = [{"role": "assistant", "content": SALUDO}]
    ss.chat_campos = {}
    ss.ficha_chat = None
    ss.post_msgs = []
    ss.interp = None
    ss.chat_version = ss.get("chat_version", 0) + 1


def finalizar_chat():
    """La ficha se entrega al sistema inteligente: a partir de aquí corre filtro + difusos + AG."""
    st.session_state.ficha_chat = perfil.ficha_desde_campos(st.session_state.chat_campos)
    st.session_state.interp = None
    st.session_state.post_msgs = []


def reinterpretar():
    st.session_state.interp = None


def pantalla_recoleccion(api_key, modelo):
    """Primer contacto: el asistente entrevista al usuario y va llenando la ficha. Termina con st.stop()."""
    ss = st.session_state
    st.title("💬 Asistente de postulación a concursos públicos")
    st.caption(f"El asistente ({asistente.proveedor(api_key)}) te entrevista y arma la ficha de tu empresa → el sistema inteligente (difuso + "
               "algoritmo genético) la evalúa → el asistente te explica el resultado.")
    if not api_key:
        st.warning("Falta la **clave de API** (Gemini u OpenRouter): pégala en la barra lateral (🤖 Asistente IA) o configúrala en "
                   "`.streamlit/secrets.toml` / variable `GEMINI_API_KEY` (ver MANUAL_INSTALACION.md). "
                   "También puedes elegir una empresa simulada en la barra lateral.")
    col_chat, col_ficha = st.columns([1.7, 1])
    error = None
    with col_chat:
        caja = st.container()
        entrada = st.chat_input("Escribe tu respuesta…", disabled=not api_key, key="entrada_recoleccion")
        if entrada and api_key:
            hist = ss.chat_msgs + [{"role": "user", "content": entrada}]
            try:
                with st.spinner("El asistente está pensando…"):
                    r = asistente.turno_recoleccion(api_key, hist, ss.chat_campos, modelo)
            except asistente.GeminiError as e:
                error = f"{e}  \nTu mensaje no se envió; vuelve a escribirlo: «{entrada}»"
            else:
                ss.chat_msgs = hist + [{"role": "assistant", "content": r["mensaje"]}]
                ss.chat_campos = r["campos"]
                if r["confirmado"]:          # ficha completa y el usuario aprobó el resumen
                    finalizar_chat()
                    st.rerun()
    with caja:
        for m in ss.chat_msgs:
            st.chat_message(m["role"]).markdown(m["content"])
        if error:
            st.error(error)
    with col_ficha, st.container(border=True):
        campos, faltan = ss.chat_campos, perfil.faltantes(ss.chat_campos)
        hechos = len(perfil.REQUERIDOS) - len(set(faltan) & set(perfil.REQUERIDOS))
        st.markdown("##### 📋 Ficha que se va armando")
        st.progress(hechos / len(perfil.REQUERIDOS), text=f"{hechos} de {len(perfil.REQUERIDOS)} datos obligatorios")
        mostrar = perfil.REQUERIDOS + (["departamentos_cobertura"] if campos.get("cobertura") == "regional" else [])
        if campos.get("nombre"):
            st.markdown(f"**Empresa:** {campos['nombre']}")
        st.markdown("\n".join(
            f"- ✅ **{perfil.ETIQUETAS[c]}:** {perfil.valor_texto(c, campos[c])}" if c in campos
            else f"- ⬜ {perfil.ETIQUETAS[c]}" for c in mostrar))
        errores = perfil.validar(perfil.ficha_desde_campos(campos)) if not faltan else []
        for e in errores:
            st.error(e)
        if not faltan and not errores:
            st.success("Ficha completa. Confirma en el chat o evalúa ahora.")
            st.button("▶ Evaluar con estos datos", type="primary", on_click=finalizar_chat)
        st.button("🔄 Empezar de nuevo", on_click=iniciar_chat)
    st.stop()


# ============================ barra lateral ============================
st.sidebar.markdown("### 🏢 Empresa y datos")
empresa_sel = st.sidebar.selectbox("Empresa", [MODO_ASISTENTE] + list(perfil.EMPRESAS),
                                   help="«Asistente»: un LLM (Gemini u OpenRouter) entrevista al usuario y arma la ficha. "
                                        "Las demás son empresas simuladas (sin IA).")
modo_chat = empresa_sel == MODO_ASISTENTE
if "chat_msgs" not in st.session_state:
    iniciar_chat()
with st.sidebar.expander("🤖 Asistente IA", expanded=modo_chat and not clave_gemini()):
    st.text_input("Clave de API (Gemini u OpenRouter)", type="password", key="gemini_key_input",
                  help="Solo se guarda en esta sesión del navegador. Mejor aún: configúrala en .streamlit/secrets.toml "
                       "o en la variable GEMINI_API_KEY.")
    st.text_input("Modelo", value=_modelo_inicial(), key=f"gemini_modelo_{asistente.proveedor(clave_gemini())}",
                  help="Gemini: por defecto gemini-flash-latest. OpenRouter (clave sk-or-…): por defecto openrouter/free, "
                       "que usa un modelo gratuito disponible. Si un modelo no existe, se prueban modelos de respaldo.")
    if clave_gemini():
        st.caption(f"Proveedor detectado: **{asistente.proveedor(clave_gemini())}**")
if modo_chat:
    st.sidebar.button("↩️ Nueva conversación", on_click=iniciar_chat)
fuente_sel = st.sidebar.radio("Fuente de concursos", ["Descarga masiva OECE", "API en vivo OECE"], horizontal=True,
                              help="Ambas son fuentes oficiales del OECE en formato OCDS.")
paginas = st.sidebar.slider("Páginas de la API (20 procesos c/u)", 5, 60, 25) if fuente_sel == "API en vivo OECE" else 25

st.sidebar.markdown("### 🧬 Modelo")
forma = st.sidebar.radio("Riesgo en el fitness", ["descuento", "resta"], horizontal=True,
                         format_func={"descuento": "Descuento", "resta": "Resta"}.get,
                         help="Descuento: Σs·(1−R/100)^λ  ·  Resta: Σs − λ·R")
lam = st.sidebar.slider("λ riesgo (aversión al riesgo)", 0.0, 6.0 if forma == "resta" else 3.0, 1.0, 0.25)
pen = st.sidebar.checkbox("Penalidades clásicas (recursos, fechas, ventana)", False,
                          help="Castigan parte de lo mismo que el riesgo difuso (doble penalización).")
with st.sidebar.expander("Parámetros avanzados"):
    umbral = st.slider("Umbral de compatibilidad (preselección)", 0, 90, 60)
    top_n = st.slider("Top-N candidatos (genes)", 5, 25, 15)
    pob = st.slider("Población", 20, 200, 80, 10)
    gens = st.slider("Generaciones máx.", 20, 400, 150, 10)
    pmut = st.slider("Prob. de mutación por gen", 0.01, 0.30, round(1 / top_n, 2), 0.01)

if modo_chat and st.session_state.ficha_chat is None:
    pantalla_recoleccion(clave_gemini(), modelo_gemini())       # primer contacto; no continúa hasta tener la ficha

try:
    df_bulk, fuente_bulk, montos_hist = cargar_bulk()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

if fuente_sel == "API en vivo OECE":
    try:
        df_src = cargar_api(paginas)
        vig, fecha_ref, corte, retro = oece.concursos_vigentes(df_src, fecha_ref=pd.Timestamp.today().normalize(),
                                                               min_vigentes=0, max_retro=0)
        fuente_txt = "API en vivo"
    except Exception as e:  # sin conexión: se vuelve a la descarga masiva
        st.sidebar.error(f"API no disponible ({e}). Se usa la descarga masiva.")
        fuente_sel = "Descarga masiva OECE"
if fuente_sel != "API en vivo OECE":
    df_src = df_bulk
    vig, fecha_ref, corte, retro = oece.concursos_vigentes(df_src)
    fuente_txt = fuente_bulk
st.sidebar.caption(f"📡 {fuente_txt} · referencia {fecha_ref.date()} · {len(df_src):,} procesos")

# ============================ encabezado ============================
nombre_empresa = ((st.session_state.ficha_chat or {}).get("nombre") or "Mi empresa") if modo_chat else empresa_sel
st.title("📊 Portafolio inteligente de concursos públicos")
st.caption(f"**{nombre_empresa}** · Difuso #1 compatibilidad → Algoritmo genético ⇄ Difuso #2 riesgo del portafolio · datos abiertos del OECE (OCDS)")
cabecera = st.container()   # se llena al final, cuando ya se ejecutó el modelo

nombres_tabs = ["🗺️ Arquitectura", "🏢 Empresa", "📥 Concursos OECE", "🎯 Compatibilidad",
                "⚠️ Riesgo del portafolio", "🧬 Algoritmo genético", "✅ Resultado"]
tabs = st.tabs(["💬 Asistente"] + nombres_tabs if modo_chat else nombres_tabs)
tab_chat = tabs[0] if modo_chat else None
if modo_chat:
    tabs = tabs[1:]          # el resto del código sigue usando tabs[0]..tabs[6]

# ============================ 0. Arquitectura ============================
with tabs[0]:
    st.graphviz_chart("""
    digraph {
      rankdir=LR; bgcolor="transparent"; nodesep=0.25; ranksep=0.35;
      node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.12,0.06"];
      edge [color="#888888", fontname="Helvetica", fontsize=8, fontcolor="#888888"];
      F [label="Asistente IA (LLM)\\n(entrevista → ficha)", fillcolor="#d7efe9"];
      O [label="Adaptador OECE\\nbulk + API en vivo", fillcolor="#d7efe9"];
      D [label="Filtro duro", fillcolor="#fff0c7"];
      C [label="Difuso #1\\nCompatibilidad\\npor concurso", fillcolor="#fbd9c9"];
      P [label="Top-N\\ncandidatos", fillcolor="#fff0c7"];
      G [label="Algoritmo genético\\nelige el portafolio", fillcolor="#d3e3f5"];
      RK [label="Difuso #2  Riesgo\\ncantidad · capital\\npersonal · fechas", fillcolor="#fbd9c9"];
      S [label="Portafolio +\\nexplicación", fillcolor="#d7efe9"];
      L [label="Asistente IA (LLM)\\n(explica el resultado)", fillcolor="#d7efe9"];
      F -> D; O -> D; D -> C -> P -> G -> S -> L;
      G -> RK [label="cada cromosoma"]; RK -> G [label="riesgo en el fitness"];
    }""", width="content")
    c1, c2, c3 = st.columns(3)
    with c1, st.container(border=True):
        st.markdown("""##### Qué hace
1. **Ficha**: perfil de la empresa; la arma el **asistente IA** (Gemini u OpenRouter) conversando con el usuario (o se llena a mano).
2. **OECE**: concursos reales convocados.
3. **Difuso #1**: compatibilidad 0-100 de **cada concurso** (6 variables, 16 reglas).
4. **AG**: busca la **combinación** de concursos con mejor fitness.
5. **Difuso #2**: riesgo 0-100 de **cada combinación** que propone el AG.
6. **Asistente**: recibe el resultado y lo **interpreta** al usuario (la IA no calcula puntajes).""")
    with c2, st.container(border=True):
        st.markdown("""##### Lo pedido: riesgo por cantidad en el fitness
`Fitness = Σ compatibilidad · (1 − Riesgo/100)^λ`

El eje principal del riesgo es **cantidad = nº licitaciones / capacidad operativa**, con un término
**EXCESIVA → riesgo CRÍTICO**. El tope duro del AG es 2× la capacidad: **la lógica difusa, no una regla fija,
decide cuántas son demasiadas**.""")
    with c3, st.container(border=True):
        st.markdown("""##### Por qué dos sistemas difusos
La compatibilidad es propiedad de **un** concurso; el riesgo es propiedad de **la combinación**.
Dos concursos seguros por separado pueden ser riesgosos juntos (suman capital y personal, y sus
cierres chocan). Por eso el riesgo se evalúa dentro del fitness, cromosoma por cromosoma.""")

# ============================ 1. Empresa ============================
with tabs[1]:
    base = st.session_state.ficha_chat if modo_chat else perfil.EMPRESAS[empresa_sel]
    k = f"chat{st.session_state.chat_version}" if modo_chat else empresa_sel  # al cambiar de empresa se recargan los valores
    c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1])
    with c1:
        kws = st.text_area("Palabras clave del rubro (una por línea)", "\n".join(base["rubro_keywords"]), height=218, key=k + "kw")
    with c2:
        cats = st.multiselect("Categorías", list(perfil.CATEGORIAS), base["categorias"], format_func=perfil.CATEGORIAS.get, key=k + "cat")
        cap_fin = st.number_input("Capacidad financiera (S/)", 10_000, 50_000_000, base["capacidad_financiera"], 50_000, key=k + "cf")
        m_min = st.number_input("Monto mínimo de interés (S/)", 0, 5_000_000, base["monto_minimo_interes"], 5_000, key=k + "mm")
    with c3:
        presup = st.number_input("Presupuesto de postulaciones (S/)", 5_000, 5_000_000, base["presupuesto_postulaciones"], 5_000, key=k + "pp")
        exp = st.number_input("Años de experiencia", 0, 60, base["experiencia_anios"], key=k + "ex")
        pers = st.number_input("Personal disponible", 1, 500, base["personal_disponible"], key=k + "pe")
    with c4:
        cap_op = st.number_input("Capacidad operativa (proyectos simultáneos)", 1, 20, base["capacidad_operativa"], key=k + "co",
                                 help="Referencia de la variable 'cantidad' del riesgo difuso.")
        dep = st.text_input("Departamento base", base["departamento_base"], key=k + "dep").upper().strip()
        cob = st.radio("Cobertura", ["nacional", "regional"], ["nacional", "regional"].index(base["cobertura"]), horizontal=True, key=k + "cob")
    deps = st.text_input("Departamentos de cobertura (separados por coma)", ", ".join(base["departamentos_cobertura"]), key=k + "deps")

    ficha = {**base,
             "rubro_keywords": [w.strip() for w in kws.splitlines() if w.strip()],
             "categorias": cats, "capacidad_financiera": cap_fin, "monto_minimo_interes": m_min,
             "presupuesto_postulaciones": presup, "experiencia_anios": exp, "personal_disponible": pers,
             "capacidad_operativa": int(cap_op), "departamento_base": dep, "cobertura": cob,
             "departamentos_cobertura": [d.strip().upper() for d in deps.split(",") if d.strip()]}
    errores = perfil.validar(ficha)
    for e in errores:
        st.error(e)
    if errores:
        st.stop()
    with st.expander("Ficha estructurada (lo que el asistente entrega al núcleo inteligente)"):
        st.json(ficha, expanded=False)

# ============================ pipeline ============================
fac, embudo, ev = evaluar_concursos(vig, ficha, montos_hist, fecha_ref)
if not len(fac):
    with cabecera:
        st.warning("Ningún concurso pasa el filtro. Amplíe palabras clave, categorías o capacidad financiera (pestaña Empresa).")
    st.stop()

cand = ev[ev["score"] >= umbral].head(top_n)
aviso_umbral = len(cand) < 3
if aviso_umbral:
    cand = ev.head(top_n)
cand = cand.drop(columns=["regla_dominante"]).reset_index(drop=True)
cand.index = [f"C{i + 1}" for i in range(len(cand))]
lambdas = {"riesgo": lam, "recursos": 1.0, "fechas": 0.3, "ventana": 0.2}
params = {"poblacion": pob, "generaciones": gens, "p_mut": pmut}
ctx, res, exh = correr_modelo(cand, ficha, fecha_ref, lambdas, forma, pen, params)
bar = correr_barrido(cand, ficha, fecha_ref, forma, pen, params)
fit = PA.Evaluador(ctx, lambdas, forma, pen)
xb, fb = res["ranking"][0]
riesgo_b = R.riesgo(xb, ctx)

# ============================ 2. Concursos ============================
with tabs[2]:
    c1, c2 = st.columns([2, 3])
    with c1:
        fig = go.Figure(go.Funnel(y=embudo["Etapa"], x=embudo["Quedan"], textinfo="value", marker=dict(color="#2a9d8f")))
        st.plotly_chart(estilo(fig, 330, "Embudo del filtro duro", leyenda=False), width="stretch")
    with c2:
        st.markdown(f"##### {len(fac)} concursos factibles · {int(fac['monto_estimado'].sum())} con monto estimado")
        st.dataframe(fac[["descripcion", "entidad", "departamento", "metodo", "monto_pen", "monto_estimado", "cierre_est"]],
                     column_config={"monto_pen": COL_MONTO, "monto_estimado": st.column_config.CheckboxColumn("estimado"),
                                    "cierre_est": st.column_config.DateColumn("cierre est."),
                                    "descripcion": st.column_config.TextColumn("descripción", width="large")},
                     height=300, hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    with c1.expander("¿Por qué un filtro duro antes de la lógica difusa?"):
        st.markdown("Hay condiciones que **no admiten grados**: otro rubro, monto sobre la capacidad financiera, "
                    "contratación directa o fuera de cobertura. No tiene sentido darles un puntaje parcial; el filtro "
                    "las descarta y la lógica difusa trabaja solo sobre lo factible.")
    with c2.expander("Dos hallazgos en los datos reales del OECE"):
        st.markdown("- `tenderPeriod` trae inicio = fin: la **vigencia** se toma de `items.statusDetails = CONVOCADO` "
                    "y el **cierre se estima** como fin de consultas + 8 días.\n"
                    "- Con la **Ley 32069** el valor estimado de bienes y servicios está **reservado** mientras está "
                    "convocado (~88 % en 0). Se **estima** con la mediana de procesos ya adjudicados del mismo método y "
                    "categoría, y se marca como estimado.")

# ============================ 3. Compatibilidad ============================
with tabs[3]:
    top = ev.head(60)
    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.markdown(f"##### Ranking individual · {len(ev)} factibles · **clic en una fila** para ver su razonamiento")
        sel = st.dataframe(top[["score", "descripcion", "monto_pen", "departamento"]],
                           column_config={"score": COL_BARRA("compat."), "monto_pen": COL_MONTO,
                                          "descripcion": st.column_config.TextColumn("descripción", width="medium"),
                                          "departamento": st.column_config.TextColumn("depto.", width="small")},
                           height=440, hide_index=True, width="stretch", on_select="rerun",
                           selection_mode="single-row", key="sel_compat")
    filas_sel = sel.selection.rows if sel and sel.selection else []
    idx = top.index[filas_sel[0]] if filas_sel else top.index[0]
    entrada = {v: float(ev.loc[idx, v]) for v in CO.ENTRADAS}
    rc = CO.SISTEMA.evaluar(entrada, detalle=True)
    with c2:
        st.markdown(f"##### {ev.loc[idx, 'descripcion'][:90]}…")
        st.dataframe(pd.DataFrame([entrada]).round(2), hide_index=True, width="stretch")
        st.plotly_chart(fig_salida(CO.SISTEMA, rc, 215, f"Salida agregada → compatibilidad {rc['valor']:.1f}"), width="stretch")
        st.dataframe(tabla_reglas(rc["activas"], "compat."), hide_index=True, width="stretch", height=145,
                     column_config={"Activación": st.column_config.ProgressColumn("Activación", min_value=0, max_value=1, format="%.2f")})
    with st.expander("Variables de entrada, funciones de pertenencia y reglas (Mamdani: min → max → centroide)"):
        grilla_mf(CO.VARIABLES, CO.UNIVERSOS, cols=6, marcas=entrada)
        a, b = st.columns([1, 1.4])
        a.dataframe(pd.DataFrame({"variable": list(CO.DESCRIPCION), "cálculo": list(CO.DESCRIPCION.values())}),
                    hide_index=True, width="stretch")
        b.dataframe(pd.DataFrame({"regla": [regla_texto(a_, c_, "compat.") for a_, c_ in CO.REGLAS]}),
                    hide_index=True, width="stretch", height=250)

# ============================ 4. Riesgo ============================
with tabs[4]:
    cap = ficha["capacidad_operativa"]
    c1, c2, c3 = st.columns([0.9, 1.3, 1.3])
    with c1, st.container(border=True):
        st.markdown("##### Simulador")
        kk = st.slider("Nº de licitaciones", 1, 2 * cap, cap, help=f"Capacidad operativa = {cap}")
        cap_r = st.slider("Capital / capacidad financiera", 0.0, 2.0, 0.3, 0.05)
        per_r = st.slider("Personal req. / disponible", 0.0, 2.0, 0.4, 0.05)
        fec_r = st.slider("Choque de cierres", 0.0, 1.0, 0.1, 0.05)
        ent_r = {"cantidad": min(2.0, kk / cap), "capital": cap_r, "personal": per_r, "fechas": fec_r}
        rr = R.SISTEMA.evaluar(ent_r, detalle=True)
        st.metric("Riesgo difuso", f"{rr['valor']:.1f} / 100", R.nivel(rr["valor"]).upper(), delta_color="off")
    with c2:
        st.plotly_chart(fig_salida(R.SISTEMA, rr, 250, "Salida agregada y centroide"), width="stretch")
        st.dataframe(tabla_reglas(rr["activas"], "riesgo"), hide_index=True, width="stretch",
                     column_config={"Activación": st.column_config.ProgressColumn("Activación", min_value=0, max_value=1, format="%.2f")})
    with c3:
        fig = go.Figure()
        for nombre, (a, b, c) in {"pequeñas": (0.05, 0.07, 0.05), "medianas": (0.1, 0.15, 0.2),
                                  "grandes": (0.2, 0.25, 0.4)}.items():
            ks, vals = R.curva_por_cantidad(cap, a, b, c)
            fig.add_trace(go.Scatter(x=ks, y=vals, mode="lines+markers", name=f"licitaciones {nombre}"))
        fig.add_vrect(x0=cap, x1=2 * cap, fillcolor="#e76f51", opacity=0.08, line_width=0,
                      annotation_text="excede capacidad", annotation_position="top left")
        fig.add_vline(x=cap, line_dash="dash", line_color=LINEA)
        fig.update_xaxes(title="nº de licitaciones", dtick=1)
        fig.update_yaxes(title="riesgo", range=[0, 100])
        st.plotly_chart(estilo(fig, 330, "📈 Riesgo vs. CANTIDAD de licitaciones"), width="stretch")
        st.caption("Cada licitación suma capital y personal. Al pasar la capacidad operativa se activa "
                   "**CANTIDAD EXCESIVA → CRÍTICO**.")
    with st.expander("Funciones de pertenencia y matriz de reglas del riesgo"):
        grilla_mf(R.VARIABLES, R.UNIVERSOS, cols=4, marcas=ent_r)
        a, b = st.columns([1, 1.3])
        a.dataframe(pd.DataFrame({"variable": list(R.DESCRIPCION), "qué mide": list(R.DESCRIPCION.values())}),
                    hide_index=True, width="stretch")
        m = pd.DataFrame(R.MATRIZ).T
        m.columns = ["base", "+capital ajust.", "+personal justo", "+fechas medio", "+fechas alto"]
        m.loc["excesiva"] = ["critico"] + ["—"] * 4
        b.dataframe(m, width="stretch")
        b.caption("Cada nivel de cantidad tiene un riesgo base y los otros factores lo escalan. "
                  "Globales: capital excedido → crítico · personal excedido → crítico.")

# ============================ 5. Algoritmo genético ============================
with tabs[5]:
    if aviso_umbral:
        st.warning(f"Menos de 3 concursos superan {umbral}; se toman los {top_n} mejores sin umbral.")
    st.caption(f"**Cromosoma:** {ctx['n']} genes binarios (1 = postular a Cᵢ) · restricción dura 1 ≤ Σx ≤ {ctx['k_max']} "
               f"(2× capacidad) y costo ≤ S/ {ctx['presupuesto']:,.0f} · espacio: **{PA.n_combinaciones(ctx):,} combinaciones** · "
               f"torneo 3, cruce uniforme, mutación {pmut:.2f}, elitismo 2, reparación · λ = {lam} ({forma}).")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Generaciones", res["generaciones"])
    c2.metric("Portafolios evaluados por el AG", f"{res['evaluaciones']:,}")
    if exh is not None:
        xe, fe, n_ev = exh
        ok = abs(fe - fb) < 1e-6
        c3.metric("Búsqueda exhaustiva", f"{n_ev:,}", "evaluados", delta_color="off")
        c4.metric("¿AG = óptimo global?", "✅ Sí" if ok else "≈ Cerca", f"óptimo {fe:.1f}", delta_color="off")
    else:
        c3.metric("Búsqueda exhaustiva", "no viable")
        c4.metric("Justificación", "usar AG")

    c1, c2 = st.columns(2)
    with c1:
        h = res["hist"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=h["mejor"], name="mejor fitness", line=dict(color="#2a9d8f")))
        fig.add_trace(go.Scatter(y=h["media"], name="promedio", line=dict(color="#e9c46a")))
        fig.add_trace(go.Scatter(y=h["k_mejor"], name="nº licitaciones (mejor)", yaxis="y2",
                                 line=dict(dash="dot", color="#e76f51")))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, ctx["k_max"] + 0.5], dtick=1,
                                      showgrid=False, title="nº licitaciones"),
                          xaxis_title="generación", yaxis_title="fitness")
        st.plotly_chart(estilo(fig, 340, "Convergencia del AG"), width="stretch")
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=bar["λ riesgo"].astype(str), y=bar["nº licitaciones"], name="nº licitaciones",
                             marker_color="#2a9d8f", text=bar["nº licitaciones"], textposition="inside"))
        fig.add_trace(go.Scatter(x=bar["λ riesgo"].astype(str), y=bar["riesgo difuso"], name="riesgo", yaxis="y2",
                                 mode="lines+markers", line=dict(color="#e76f51")))
        fig.update_layout(xaxis=dict(title="λ (aversión al riesgo)", type="category"),
                          yaxis=dict(title="nº licitaciones", dtick=1),
                          yaxis2=dict(overlaying="y", side="right", range=[0, 100], showgrid=False, title="riesgo"))
        st.plotly_chart(estilo(fig, 340, "🎚️ Barrido de λ: más aversión al riesgo → menos licitaciones"), width="stretch")
    c1, c2 = st.columns(2)
    with c1.expander("Tabla del barrido de λ"):
        st.dataframe(bar, hide_index=True, width="stretch")
    with c2.expander("¿Resta o descuento?"):
        st.markdown("En la **resta** (`Σs − λ·R`), Σs crece sin límite con cada licitación, pero R se satura en 100: "
                    "el efecto de λ es todo o nada. En el **descuento** (`Σs·(1−R/100)^λ`, valor esperado), una licitación "
                    "extra solo conviene si su aporte supera lo que el riesgo le quita a **todo** el portafolio, y la "
                    "cantidad baja de forma gradual. Se pueden comparar las dos formas desde la barra lateral.")

# ============================ 6. Resultado ============================
with tabs[6]:
    filas = []
    for i, (x, fx) in enumerate(res["ranking"][:5], 1):
        d = fit.descomposicion(x)
        filas.append({"opción": f"#{i}", "concursos": "+".join(cand.index[x == 1]), "nº": int(x.sum()),
                      **{k2: round(v, 1) for k2, v in d.items()}, "riesgo": round(R.riesgo(x, ctx), 1),
                      "fitness": round(fx, 1)})
    alt = pd.DataFrame(filas)
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.markdown("##### Top 5 portafolios")
        st.dataframe(alt, hide_index=True, width="stretch",
                     column_config={"opción": st.column_config.TextColumn("#", width="small"),
                                    "nº": st.column_config.NumberColumn("nº", width="small"),
                                    "concursos": st.column_config.TextColumn("concursos", width="small"),
                                    "Σ compatibilidad": st.column_config.NumberColumn("Σ compat.", format="%.1f"),
                                    "− pérdida por riesgo": st.column_config.NumberColumn("− riesgo", format="%.1f"),
                                    "− λ·riesgo": st.column_config.NumberColumn("− λ·R", format="%.1f"),
                                    "riesgo": COL_BARRA("riesgo"),
                                    "fitness": st.column_config.NumberColumn("fitness", format="%.1f")})
        t1, t2 = st.tabs(["📊 Descomposición del fitness", "📅 Calendario del recomendado"])
        with t1:
            comp = alt.melt(id_vars="opción", value_vars=[c for c in alt.columns if c[0] in "Σ−"],
                            var_name="término", value_name="aporte")
            fig = px.bar(comp, x="opción", y="aporte", color="término", barmode="relative",
                         color_discrete_sequence=COLORES)
            fig.update_xaxes(type="category", title=None)
            st.plotly_chart(estilo(fig, 245), width="stretch")
        with t2:
            g = pd.DataFrame([{"concurso": cid, "fase": "preparación", "inicio": fecha_ref, "fin": ctx["cierre"][i]}
                              for i, cid in enumerate(cand.index) if xb[i]] +
                             [{"concurso": cid, "fase": "ejecución", "inicio": ctx["ejec_ini"][i], "fin": ctx["ejec_fin"][i]}
                              for i, cid in enumerate(cand.index) if xb[i]])
            fig = px.timeline(g, x_start="inicio", x_end="fin", y="concurso", color="fase",
                              color_discrete_sequence=["#e9c46a", "#2a9d8f"])
            st.plotly_chart(estilo(fig, 245), width="stretch")
    with c2, st.container(border=True, height=545):
        st.markdown(explicacion.texto(xb, cand, ctx, fit, ficha))

# ============================ cabecera (resumen siempre visible) ============================
with cabecera:
    cols = st.columns(6)
    cols[0].metric("Concursos vigentes", f"{len(vig):,}")
    cols[1].metric("Factibles para la empresa", f"{len(fac):,}")
    cols[2].metric("Candidatos al AG (genes)", ctx["n"])
    cols[3].metric("Licitaciones / capacidad", f"{int(xb.sum())} de {ficha['capacidad_operativa']}")
    cols[4].metric("Riesgo del portafolio", f"{riesgo_b:.0f} / 100", R.nivel(riesgo_b), delta_color="off")
    cols[5].metric("Fitness", f"{fb:.1f}", f"λ {lam} · {forma}", delta_color="off")

# ============================ Asistente: interpretación del resultado ============================
if modo_chat:
    with tab_chat:
        ss = st.session_state
        api_key, modelo = clave_gemini(), modelo_gemini()
        estado = repr((round(float(lam), 2), forma, bool(pen), tuple(int(v) for v in xb), sorted(ficha.items(), key=str)))

        def resultados_json():
            return asistente.contexto_resultados(ficha, cand, xb, res["ranking"], ctx, fit, embudo, bar, lam, forma)

        if ss.get("interp") is None:      # se interpreta UNA vez (no en cada movimiento de un slider: cuida la cuota gratuita)
            try:
                if not api_key:
                    raise asistente.GeminiError("No hay clave de API configurada.", "sin_clave")
                with st.spinner("El asistente está interpretando los resultados…"):
                    texto, usado = asistente.interpretar(api_key, resultados_json(), modelo=modelo)
                ss.interp = {"estado": estado, "ok": True, "modelo": usado}
            except asistente.GeminiError as e:
                texto = explicacion.texto(xb, cand, ctx, fit, ficha)
                ss.interp = {"estado": estado, "ok": False, "error": str(e)}
            ss.post_msgs = [{"role": "assistant", "content": texto}]

        if not ss.interp["ok"]:
            st.warning(f"No pude usar el asistente IA ({ss.interp['error']}). Te muestro la explicación estándar del sistema.")
        c1, c2 = st.columns([4, 1])
        if ss.interp["estado"] != estado:
            c1.info("Los parámetros o el resultado cambiaron desde esta explicación. Pulsa «Reinterpretar» para actualizarla.")
        else:
            c1.caption(f"Explicación generada por **{ss.interp.get('modelo', 'el sistema')}** con λ = {lam} · {forma}. "
                       "El asistente solo interpreta: los cálculos los hizo el sistema inteligente.")
        c2.button("🔄 Reinterpretar", on_click=reinterpretar, disabled=not api_key)

        caja = st.container()
        pregunta = st.chat_input("Pregúntale al asistente sobre estos resultados…", disabled=not api_key,
                                 key="entrada_resultados")
        error = None
        if pregunta and api_key:
            hist = ss.post_msgs + [{"role": "user", "content": pregunta}]
            try:
                with st.spinner("El asistente está pensando…"):
                    texto, _ = asistente.interpretar(api_key, resultados_json(), historial=hist, modelo=modelo)
                ss.post_msgs = hist + [{"role": "assistant", "content": texto}]
            except asistente.GeminiError as e:
                error = f"{e}  \nTu pregunta no se envió; vuelve a escribirla: «{pregunta}»"
        with caja:
            for m in ss.post_msgs:
                st.chat_message(m["role"]).markdown(m["content"])
            if error:
                st.error(error)
        with st.expander("Ver la conversación inicial (recolección de datos)"):
            for m in ss.chat_msgs:
                st.chat_message(m["role"]).markdown(m["content"])
