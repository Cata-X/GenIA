"""Demo para el video (<= 3 min): un laberinto NUEVO, baseline vs sistema multiagente, en vivo.

    python -m src.demo                      # semilla al azar
    python -m src.demo --semilla 12345      # repetir la misma demo
    python -m src.demo --llm simulado       # ensayo sin GPU (NO usar en el video)

Imprime los mensajes entre agentes y herramientas a medida que ocurren y guarda la traza completa
(con los prompts) en traces/demo_<semilla>.jsonl. No hay montaje: lo que se ve es la corrida real.
"""
import argparse
import random
from pathlib import Path

from .maze import generar_laberinto, bfs, dibujar
from .d2_legacy import correr_baseline
from .orchestrator import CONFIGS, Tracer, resolver
from .evaluate import RAIZ


def demo(llm, semilla=None, config="completo", con_baseline=True, traces_dir=None):
    semilla = semilla or random.randint(10_000, 99_999)
    lab, a, b = generar_laberinto(semilla)
    print(f"Semilla de la demo: {semilla} | inicio {a} | meta {b} | óptimo {len(bfs(lab, a, b)) - 1} pasos\n")
    dibujar(lab, [], a, b)
    if con_baseline:
        print("\n========== BASELINE (direct prompting) ==========")
        rb = correr_baseline(llm.como_funcion_d2(), lab, a, b)
        print(rb["salida_cruda"][:700])
        print("\nVEREDICTO BASELINE:", rb["estado"])
    print(f"\n========== SISTEMA MULTIAGENTE ({config}) ==========")
    path = Path(traces_dir or RAIZ / "traces") / f"demo_{semilla}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    tr = Tracer(path=path, verbose=True, guardar_prompts=True)
    r = resolver(llm, lab, a, b, CONFIGS[config], id_lab=semilla, tracer=tr)
    tr.volcar()
    print("\nVEREDICTO SISTEMA:", r["estado"], "| pasos:", r["pasos"], "| rondas sin candidato:", r["contadores"]["rondas_vacias"])
    dibujar(lab, r["ruta"], a, b)
    print(f"\nTraza completa: {path}")
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--semilla", type=int, default=None)
    ap.add_argument("--config", default="completo", choices=list(CONFIGS))
    ap.add_argument("--llm", choices=["qwen", "simulado"], default="qwen")
    ap.add_argument("--sin-baseline", action="store_true")
    a = ap.parse_args(argv)
    if a.llm == "qwen":
        from .llm import QwenLLM
        llm = QwenLLM()
    else:
        from .llm_simulado import LLMSimulado
        llm = LLMSimulado()
    demo(llm, a.semilla, a.config, not a.sin_baseline)


if __name__ == "__main__":
    main()
