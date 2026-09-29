"""Descarga y procesa UNA vez los datos abiertos del OECE, y genera la muestra de respaldo.

Uso:  python scripts/preparar_datos.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import oece  # noqa: E402

t0 = time.time()
print("Descargando", oece.URL_BULK)
oece.descargar_bulk(lambda p: print(f"\r{p:.0%}", end=""))
print("\nProcesando JSON Lines (puede tardar unos minutos)...")
df = oece.procesar_bulk()
print(f"{len(df):,} procesos -> {oece.CACHE}  ({time.time() - t0:.0f} s)")

# Muestra de respaldo: los procesos vigentes + recientes, para que la demo funcione sin internet
vig, fref, corte, retro = oece.concursos_vigentes(df)
recientes = df[df["fecha_publicacion"] >= fref - __import__("pandas").Timedelta(days=60)]
muestra = df.loc[vig.index.union(recientes.index)]
muestra.to_parquet(oece.MUESTRA, index=False)
print(f"Vigentes: {len(vig)} (ref {fref.date()}, corte {corte.date()}, retroceso {retro} d)")
print(f"Muestra de respaldo: {len(muestra):,} procesos -> {oece.MUESTRA}")
