"""Evaluacion: mismos 20 laberintos y mismo verificador (`evaluar_ruta`) para todos los metodos.

Metodos
    aleatorio, baseline, d2_agente      referencias del Entregable 2 (mismo codigo, ver d2_legacy.py)
    completo                            sistema multiagente con todas las tecnicas
    sin_fewshot, sin_observar,          ablaciones: cada una apaga UNA tecnica
    sin_selfconsistency, sin_critico,
    sin_arbitro
    arbitro_regla                       extra: el criterio del Arbitro aplicado en codigo, sin LLM

Es reanudable: cada fila (id, metodo) se agrega al CSV apenas termina, y lo ya hecho se salta.
Las trazas de mensajes de cada corrida van a traces/<metodo>.jsonl.

Uso (desde la raiz del repo):
    python -m src.evaluate --llm qwen
    python -m src.evaluate --llm simulado --n 3 --out /tmp/prueba     # ensayo sin GPU
    python -m src.evaluate --importar-d2 ruta/a/resultados.csv        # reutilizar filas del D2
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

from .maze import construir_dataset, bfs, correr_aleatorio
from .d2_legacy import correr_baseline, correr_agente
from .orchestrator import CONFIGS, Tracer, resolver

RAIZ = Path(__file__).resolve().parent.parent
REFERENCIAS = ["aleatorio", "baseline", "d2_agente"]
ORDEN = REFERENCIAS + list(CONFIGS)

COLUMNAS_D2 = ["id", "metodo", "estado", "exito", "exito_estricto", "pasos", "optimo", "invalidas",
               "valor_mal_declarado", "coordenada_mal_declarada", "tiempo_s"]
COLUMNAS_D3 = ["propuestas", "rondas_vacias", "llamadas_navegante", "llamadas_critico", "llamadas_arbitro",
               "critico_revisiones", "critico_aciertos", "critico_correcciones", "critico_rechazos",
               "critico_ilegibles", "critico_errores_graves",
               "arbitro_decisiones", "arbitro_cambia_voto", "arbitro_fallbacks",
               "version_prompts", "modelo", "ruta"]
COLUMNAS = COLUMNAS_D2 + COLUMNAS_D3


# ============================================================================ datos
def cargar_dataset(ruta=None, N=20):
    """Lee data/laberintos_9x9.json (los 20 laberintos del D2). Si no existe, los regenera con la
    misma semilla del D2 y lo guarda."""
    ruta = Path(ruta or RAIZ / "data" / "laberintos_9x9.json")
    datos = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else []
    if len(datos) < N:                       # el laberinto i usa la semilla 2026+i: los N primeros no cambian
        datos = construir_dataset(N)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(json.dumps(datos), encoding="utf-8")
    return datos[:N]


# ============================================================================ ejecucion de un metodo
def correr_metodo(llm, metodo, x, traza_path=None, verbose=False, guardar_prompts=False):
    """Devuelve (dict de resultado con las claves del D2, tracer o None)."""
    lab, a, b = x["laberinto"], tuple(x["inicio"]), tuple(x["meta"])
    if metodo == "aleatorio":
        return correr_aleatorio(lab, a, b, seed=x["id"]), None
    if metodo in ("baseline", "d2_agente"):
        gen = llm.como_funcion_d2()
        r = correr_baseline(gen, lab, a, b) if metodo == "baseline" else correr_agente(gen, lab, a, b, usar_memoria=True)
        return r, None
    cfg = CONFIGS[metodo]
    tr = Tracer(path=traza_path, verbose=verbose, guardar_prompts=guardar_prompts)
    return resolver(llm, lab, a, b, cfg, id_lab=x["id"], tracer=tr), tr


def fila_de(x, metodo, r, modelo=""):
    lab, a, b = x["laberinto"], tuple(x["inicio"]), tuple(x["meta"])
    exito = r["estado"] == "exito"
    c = r.get("contadores", {})
    fila = {"id": x["id"], "metodo": metodo, "estado": r["estado"], "exito": exito,
            "exito_estricto": r.get("exito_estricto", exito and not r.get("invalidas")),
            "pasos": r["pasos"], "optimo": len(bfs(lab, a, b)) - 1, "invalidas": r["invalidas"],
            "valor_mal_declarado": r["valor_mal_declarado"],
            "coordenada_mal_declarada": r["coordenada_mal_declarada"], "tiempo_s": r["tiempo_s"],
            "version_prompts": CONFIGS[metodo].version_prompts if metodo in CONFIGS else "",
            "modelo": modelo if metodo != "aleatorio" else "", "ruta": json.dumps(r["ruta"])}
    for k in COLUMNAS_D3:
        if k in c:
            fila[k] = c[k]
    return fila


def _agregar(csv_path, fila):
    nuevo = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS, extrasaction="ignore")
        if nuevo:
            w.writeheader()
        w.writerow(fila)


def hechos_previos(csv_path):
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {(int(r["id"]), r["metodo"]) for r in csv.DictReader(f)}


# ============================================================================ bucle principal
def evaluar(llm, metodos=None, N=20, out_dir=None, traces_dir=None, dataset=None, modelo="", verbose=False):
    """Corre `metodos` sobre los N laberintos (laberinto por laberinto, para que una corrida parcial
    ya sea comparable entre metodos). Reanudable."""
    metodos = metodos or ORDEN
    out_dir = Path(out_dir or RAIZ / "results")
    traces_dir = Path(traces_dir or RAIZ / "traces")
    out_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "resultados_d3.csv"
    datos = cargar_dataset(dataset, N)
    hechos = hechos_previos(csv_path)
    t0 = time.time()
    for x in datos:
        for metodo in metodos:
            if (x["id"], metodo) in hechos:
                continue
            tpath = traces_dir / f"{metodo}.jsonl"
            r, tr = correr_metodo(llm, metodo, x, traza_path=tpath, verbose=verbose)
            if tr is not None:
                tr.volcar()
            else:                                  # referencias: se guarda la salida cruda
                with open(tpath, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"id": x["id"], "metodo": metodo, "ruta": r["ruta"],
                                        "salida_cruda": r.get("salida_cruda")}, ensure_ascii=False, default=str) + "\n")
            _agregar(csv_path, fila_de(x, metodo, r, modelo))
            print(f"lab {x['id']:>2} | {metodo:<20} | {r['estado']:<20} | {r['pasos']:>3} pasos | {r['tiempo_s']} s"
                  f" | total {time.time() - t0:,.0f} s", flush=True)
    return csv_path


def importar_d2(csv_d2, out_dir=None, mapeo=None):
    """Copia al CSV del D3 las filas del D2 (baseline, agente, aleatorio) para no volver a correrlas.
    El D2 llama 'agente' a lo que aqui se llama 'd2_agente'."""
    import pandas as pd
    mapeo = mapeo or {"baseline": "baseline", "agente": "d2_agente", "aleatorio": "aleatorio"}
    out_dir = Path(out_dir or RAIZ / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "resultados_d3.csv"
    hechos = hechos_previos(csv_path)
    df = pd.read_csv(csv_d2)
    n = 0
    for _, r in df.iterrows():
        metodo = mapeo.get(r["metodo"])
        if metodo is None or (int(r["id"]), metodo) in hechos:
            continue
        fila = {k: r[k] for k in COLUMNAS_D2 if k != "metodo" and k in r}
        fila["metodo"] = metodo
        fila["modelo"] = "importado del D2"
        _agregar(csv_path, fila)
        n += 1
    print(f"Importadas {n} filas del D2 a {csv_path}")
    return csv_path


# ============================================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", choices=["qwen", "simulado"], default="qwen")
    ap.add_argument("--metodos", nargs="*", default=None, help=f"por defecto todos: {ORDEN}")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default=None, help="carpeta de resultados (default: results/)")
    ap.add_argument("--traces", default=None, help="carpeta de trazas (default: traces/)")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--importar-d2", default=None, help="CSV de resultados del D2 a reutilizar")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    if a.importar_d2:
        importar_d2(a.importar_d2, a.out)
    if a.llm == "qwen":
        from .llm import QwenLLM, MODEL_NAME, MODEL_REVISION
        llm, modelo = QwenLLM(), f"{MODEL_NAME}@{MODEL_REVISION}"
    else:
        from .llm_simulado import LLMSimulado
        llm, modelo = LLMSimulado(), "SIMULADO (no es el modelo)"
    evaluar(llm, a.metodos, a.n, a.out, a.traces, a.dataset, modelo, a.verbose)


if __name__ == "__main__":
    main()
