# Entregable 3 — Sistema multiagente para navegar laberintos 9x9 (Grupo 2, Generative AI 580694)

Tres agentes —**Navegante**, **Crítico-Corrector** y **Árbitro**— que usan el *mismo* modelo abierto
(**Qwen2.5-1.5B-Instruct**, 1.5 B parámetros) y se reparten el trabajo de cada paso:
**generar** varias propuestas, **validar** cada una contra hechos del entorno y **elegir** entre las válidas.
Se evalúa en los mismos 20 laberintos del Entregable 2 y con el mismo verificador (`evaluar_ruta`).

> **Estado (léelo antes de reportar resultados):** el código y los prompts están probados con un
> modelo simulado y con pruebas de formato (`pytest`, 42 pruebas). **Todavía no se ha corrido con
> Qwen real**: no hay números de este sistema. El primer paso es la prueba de humo (ver más abajo).

## Estructura

```
prompts/            un archivo por agente, versionado (navegante_v1, critico_v1, arbitro_v1, ...) + README con formatos
src/
  maze.py           laberintos, BFS, verificador evaluar_ruta   (copiado sin cambios del D2)
  d2_legacy.py      baseline y agente del D2                     (copiado sin cambios del D2)
  tools.py          herramientas: observar(), validar(), memoria (visitadas/ruta), mover()
  prompts.py        carga de prompts + transformaciones de ablación
  agents.py         Navegante, Crítico, Árbitro y sus parsers
  orchestrator.py   flujo del sistema, Config/CONFIGS (ablaciones), trazas
  llm.py            QwenLLM (greedy y muestreo)      llm_simulado.py  modelo simulado para pruebas sin GPU
  evaluate.py       corre los métodos sobre los 20 laberintos (reanudable)   analisis.py  tablas, IC, McNemar
  demo.py           corrida en vivo para el video
data/laberintos_9x9.json   los 20 laberintos (id 0 = el del D1; id i = semilla 2026+i)
results/            resultados_d3.csv, resumen.csv
traces/             <método>.jsonl con los mensajes entre agentes; caso_fallo.jsonl
notebooks/Entregable3_GenAI.ipynb   cuaderno de Colab que ejecuta todo
tests/              pruebas (sin GPU)
```

## Cómo reproducir (Colab, GPU T4)

1. Abrir `notebooks/Entregable3_GenAI.ipynb` en Colab → *Entorno de ejecución → T4 GPU* → ejecutar en orden.
2. Equivalente por consola, desde la raíz del repo:
   ```bash
   pip install -r requirements.txt
   python -m pytest -q tests                         # sin GPU, segundos
   python -m src.evaluate --n 3 --metodos completo   # prueba de humo (3 laberintos)
   python -m src.evaluate                            # los 10 métodos x 20 laberintos (reanudable)
   python -m src.analisis --caso-fallo               # tablas + traza del caso de fallo
   python -m src.demo --semilla 12345                # corrida en vivo para el video
   ```
   El modelo se fija por hash de revisión (`src/llm.py`), igual que en el D2. Sin GPU: `--llm simulado`
   ejecuta todo el pipeline con un modelo falso (**solo para probar el código; esos números no se reportan**).
3. Para reutilizar el baseline y el agente del D2 sin volver a correrlos:
   `python -m src.evaluate --importar-d2 ruta/resultados.csv` (el D2 llama `agente` a lo que aquí es `d2_agente`).

## Técnicas, dónde viven y cómo se ablacionan

| Técnica (bloque de la Clase 5) | Dónde | Flag / método |
|---|---|---|
| Prompting: few-shot positivo (5 ejemplos), rol, etiquetas XML | `prompts/navegante_v1.txt` | `sin_fewshot` |
| Context engineering — *select*: `observar()` entrega vecinos con coordenada y marca de visitada | `tools.py` | `sin_observar` |
| Context engineering — *write*: memoria `visitadas`/`ruta` | `tools.py` | (siempre activa; su efecto ya se midió en el D2) |
| Reasoning: self-consistency (k=3, T=0.7, mayoría) | `orchestrator.py` | `sin_selfconsistency` (k=1, greedy) |
| Structure / Self-Refine: Crítico con hechos del entorno + reintento con el hecho corregido | `critico_v1.txt`, `orchestrator.py` | `sin_critico` |
| Selección entre candidatos válidos (Árbitro) | `arbitro_v1.txt` | `sin_arbitro` (mayoría de votos) · extra `arbitro_regla` (mismos criterios en código) |
| Context engineering — *isolate*: cada agente ve solo su contexto; el reintento no reinyecta la salida errónea | `orchestrator.py` | (no se ablaciona por separado) |

Cada método apaga **una** técnica y deja lo demás igual a `completo`.

## Decisiones y límites que conviene declarar en el PDF

- **`sin_selfconsistency` también deja al Árbitro sin trabajo** (con k=1 siempre hay un solo candidato), así que mezcla ambos efectos; `sin_arbitro` y `arbitro_regla` aíslan al Árbitro.
- **Sin Crítico** el nombre del movimiento decide (convención del D2) y la herramienta solo comprueba legalidad; hay reintento con el motivo generado por la herramienta. Con Crítico, si la coordenada es válida manda la coordenada (regla del prompt). Es una hipótesis: mirar en la prueba de humo con cuál de las dos acierta más el modelo.
- Si el Crítico responde fuera de formato **se abstiene** y manda el nombre (`critico_ilegibles` en el CSV). Si aprueba o corrige hacia una celda ilegal, `mover()`/la herramienta lo frena (`critico_errores_graves`).
- Todo movimiento ejecutado pasa por `mover()`; por eso las rutas del sistema nunca pisan muro ni saltan. El éxito lo decide igualmente `evaluar_ruta`.
- `exito_estricto` = éxito sin ninguna ronda sin candidatos. No es comparable con el `exito_estricto` del D2 (que cuenta cualquier propuesta inválida).
- Muestreo: temperatura pura (top_p=1, top_k=0) con semilla fija por (laberinto, paso, ronda). `repetition_penalty=1.1` de `generation_config.json` de Qwen sigue activa, como en el D2.
- El Crítico se llama en lote (padding a la izquierda); en fp16 puede diferir en algún token de la llamada individual.
- Las llamadas al modelo por paso son ~1 (Navegante, k muestras) + hasta 3 (Crítico) + 1 (Árbitro): el sistema es varias veces más lento que el agente del D2. Medir tiempo en la prueba de humo antes de lanzar las 10 × 20 corridas.

## Checklist del entregable

- [ ] Prueba de humo en T4 (3 laberintos, `--verbose`): ¿el Crítico acierta?, ¿el Navegante copia la coordenada?, ¿cuánto tarda? Si cambia un prompt, crear `_v2` (no editar `_v1` después de correr resultados).
- [ ] Corrida completa → `results/resultados_d3.csv` y `traces/*.jsonl`; `python -m src.analisis --caso-fallo`.
- [ ] Verificar que `data/laberintos_9x9.json` es idéntico al `laberintos_9x9.json` del D2 (carpeta `GenIA_Entregable2` en Drive).
- [ ] Video ≤ 3 min con `python -m src.demo` (corrida real, sin montaje) y caso de fallo guardado en `traces/`.
- [ ] PDF de 1 página (vertical, LaTeX): grafo de agentes y herramientas con aristas rotuladas, bucle y condiciones de corte.
- [ ] Repo en GitHub con este README, `prompts/` y los resultados.
