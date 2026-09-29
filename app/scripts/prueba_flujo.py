"""Prueba de punta a punta del flujo (sin interfaz)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import compatibilidad, filtro, oece, perfil, portafolio_ag as PA, riesgo as R  # noqa: E402

df, fuente = oece.cargar_bulk()
vig, fref, corte, retro = oece.concursos_vigentes(df)
mh = oece.tabla_montos_historicos(df)
print(fuente, "| vigentes:", len(vig), "| ref", fref.date())

for nombre, ficha in perfil.EMPRESAS.items():
    print("\n=====", nombre)
    fac, embudo = filtro.aplicar(vig, ficha, mh)
    print(embudo.to_string(index=False))
    ev = compatibilidad.evaluar(fac, ficha, fref)
    print(ev[["descripcion", "departamento", "monto_pen", "monto_estimado", "score"]].head(5).to_string())
    cand = ev[ev.score >= 60].head(15)
    if len(cand) < 5:
        cand = ev.head(15)
    cand = cand.reset_index(drop=True)
    ctx = PA.construir_contexto(cand, ficha, fref)
    print("N =", ctx["n"], "k_max =", ctx["k_max"], "combinaciones =", PA.n_combinaciones(ctx))
    print("curva riesgo vs k:", [round(v) for v in R.curva_por_cantidad(ficha["capacidad_operativa"])[1]])
    t = time.time()
    fit = PA.Evaluador(ctx, {"riesgo": 1.0})
    res = PA.algoritmo_genetico(ctx, fit)
    x = res["ranking"][0][0]
    xb, fb, n_ev = PA.exhaustivo(ctx, PA.Evaluador(ctx, {"riesgo": 1.0}))
    print(f"AG: k={x.sum()} fit={res['ranking'][0][1]:.1f} gens={res['generaciones']} evals={res['evaluaciones']} "
          f"| exhaustivo fit={fb:.1f} ({n_ev} comb) | {time.time()-t:.1f}s")
    print("RESTA");print(PA.barrido_lambda(ctx, {}, [0, 1, 2, 3, 4, 6], "resta").to_string(index=False))
    print("DESCUENTO");print(PA.barrido_lambda(ctx, {}, [0, 0.25, 0.5, 1, 1.5, 2, 3]).to_string(index=False))
