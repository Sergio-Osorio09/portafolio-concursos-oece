"""Explicación final en texto (simula la respuesta que daría el chatbot)."""
from . import riesgo as R


def texto(x, cand, ctx, fit, ficha):
    sel = cand[x == 1]
    r = R.riesgo(x, ctx, detalle=True)
    ent = r["entradas"]
    d = fit.descomposicion(x)
    lineas = [f"**Portafolio recomendado: {int(x.sum())} licitaciones** "
              f"(capacidad operativa declarada: {ficha['capacidad_operativa']} proyectos simultáneos).", ""]
    for cid, f in sel.iterrows():
        est = " (monto estimado)" if f.get("monto_estimado") else ""
        lineas.append(f"- **{cid}** · {f['descripcion'][:110]} — {f['entidad'][:45]} · S/ {f['monto_pen']:,.0f}{est} · "
                      f"compatibilidad {f['score']:.0f}/100")
    lineas += [
        "",
        f"**Por qué:** suma de compatibilidad {d['Σ compatibilidad']:.0f} puntos; "
        f"riesgo difuso del portafolio **{r['valor']:.0f}/100 ({R.nivel(r['valor'])})**.",
        f"- Cantidad: {int(x.sum())} de {ficha['capacidad_operativa']} → {ent['cantidad']:.0%} de la capacidad operativa.",
        f"- Capital comprometido: {ent['capital']:.0%} de la capacidad financiera.",
        f"- Personal requerido: {ent['personal']:.0%} del disponible.",
        f"- Choque de cierres de postulación: {ent['fechas']:.2f} (0 = cierres separados, 1 = mismo día).",
    ]
    if r["activas"]:
        from .difuso import regla_texto
        lineas.append(f"- Regla de riesgo dominante: *{regla_texto(*r['activas'][0][:2], salida='riesgo')}* "
                      f"(activación {r['activas'][0][2]:.2f}).")
    if ent["personal"] > 1:
        lineas.append("\n⚠️ La cartera excede el personal disponible: considerar apoyo temporal.")
    lineas.append("\n*Limitación: montos reservados, personal, duración y fechas de cierre son estimaciones; "
                  "revisar las bases de cada concurso en el SEACE.*")
    return "\n".join(lineas)
