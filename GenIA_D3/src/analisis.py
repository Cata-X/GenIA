"""Analisis de results/resultados_d3.csv: tasas de exito con IC, comparacion pareada contra `completo`
(McNemar exacto, mismos 20 laberintos), taxonomia de fallos, metricas del Critico y caso de fallo.

    python -m src.analisis                       # imprime y escribe results/resumen.csv
    python -m src.analisis --caso-fallo          # copia una traza de fallo de `completo` a traces/caso_fallo.jsonl
"""
import argparse
import json
import math
from pathlib import Path

import pandas as pd

from .evaluate import ORDEN, RAIZ


def wilson(k, n, z=1.96):
    """Intervalo de Wilson al 95 % para una proporcion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exacto(base, otro):
    """base, otro: Series booleanas indexadas por id. Devuelve (b, c, p) con
    b = base acierta y otro falla, c = base falla y otro acierta, p bilateral (binomial exacto)."""
    from scipy.stats import binomtest
    j = pd.concat([base, otro], axis=1, keys=["a", "b"]).dropna().astype(bool)
    b = int((j.a & ~j.b).sum())
    c = int((~j.a & j.b).sum())
    p = 1.0 if b + c == 0 else float(binomtest(min(b, c), b + c, 0.5).pvalue)
    return b, c, p


def cargar(csv_path=None):
    csv_path = Path(csv_path or RAIZ / "results" / "resultados_d3.csv")
    df = pd.read_csv(csv_path)
    df["exito"] = df["exito"].astype(bool)
    df["exito_estricto"] = df["exito_estricto"].astype(bool)
    return df


def resumen(df, referencia="completo"):
    filas = []
    ref = df[df.metodo == referencia].set_index("id")["exito"] if referencia in set(df.metodo) else None
    for m in [m for m in ORDEN if m in set(df.metodo)]:
        d = df[df.metodo == m]
        n, k = len(d), int(d.exito.sum())
        lo, hi = wilson(k, n)
        ok = d[d.exito]
        fila = {"metodo": m, "n": n, "exitos": k, "tasa": k / n, "ic95_inf": lo, "ic95_sup": hi,
                "tasa_estricta": d.exito_estricto.mean(),
                "pasos_sobre_optimo_en_exitos": (ok.pasos / ok.optimo).mean() if len(ok) else float("nan"),
                "tiempo_prom_s": d.tiempo_s.mean(),
                "llamadas_modelo_prom": (d[["llamadas_navegante", "llamadas_critico", "llamadas_arbitro"]].sum(axis=1).mean()
                                         if d.llamadas_navegante.notna().any() else float("nan"))}
        if ref is not None and m != referencia:
            b, c, p = mcnemar_exacto(ref, d.set_index("id")["exito"])
            fila.update({"completo_gana": b, "completo_pierde": c, "p_mcnemar": p})
        filas.append(fila)
    return pd.DataFrame(filas)


def taxonomia(df):
    return pd.crosstab(df.metodo, df.estado).reindex([m for m in ORDEN if m in set(df.metodo)])


def critico(df):
    d = df[df.critico_revisiones.fillna(0) > 0]
    if d.empty:
        return pd.DataFrame()
    g = d.groupby("metodo")[["critico_revisiones", "critico_aciertos", "critico_correcciones",
                              "critico_rechazos", "critico_ilegibles", "critico_errores_graves",
                              "propuestas", "invalidas", "coordenada_mal_declarada"]].sum()
    g["acierto_critico"] = g.critico_aciertos / g.critico_revisiones
    g["prop_coord_mal"] = g.coordenada_mal_declarada / g.propuestas
    g["prop_invalidas"] = g.invalidas / g.propuestas
    return g


def caso_fallo(df, traces_dir=None, metodo="completo"):
    """Copia a traces/caso_fallo.jsonl la traza del primer laberinto donde `metodo` no llega a la meta."""
    traces_dir = Path(traces_dir or RAIZ / "traces")
    f = df[(df.metodo == metodo) & ~df.exito]
    if f.empty:
        print(f"`{metodo}` resolvio todos los laberintos: no hay caso de fallo que guardar.")
        return None
    id_ = int(f.id.iloc[0])
    destino = traces_dir / "caso_fallo.jsonl"
    n = 0
    with open(traces_dir / f"{metodo}.jsonl", encoding="utf-8") as src, open(destino, "w", encoding="utf-8") as out:
        for linea in src:
            ev = json.loads(linea)
            if ev.get("id") == id_:
                out.write(linea); n += 1
    print(f"Caso de fallo: laberinto {id_} ({f.estado.iloc[0]}), {n} eventos -> {destino}")
    return id_


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--caso-fallo", action="store_true")
    ap.add_argument("--metodo-fallo", default="completo")
    a = ap.parse_args(argv)
    df = cargar(a.csv)
    pd.set_option("display.width", 200)
    r = resumen(df)
    print(f"N = {df.id.nunique()} laberintos | metodos: {', '.join(r.metodo)}\n")
    print(r.round(3).to_string(index=False))
    print("\nTipo de resultado por metodo:\n", taxonomia(df).to_string())
    c = critico(df)
    if not c.empty:
        print("\nCritico (sumas sobre todas las corridas):\n", c.round(3).to_string())
    out = Path(a.csv).parent if a.csv else RAIZ / "results"
    r.to_csv(out / "resumen.csv", index=False)
    print(f"\nEscrito: {out / 'resumen.csv'}")
    if a.caso_fallo:
        caso_fallo(df, metodo=a.metodo_fallo)


if __name__ == "__main__":
    main()
