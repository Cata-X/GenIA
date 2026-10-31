# Prompts v1 — Entregable 3 (Grupo 2)

Un archivo por agente, fuera del código y versionado. Regla de versiones: **no se edita un archivo `_v1` después de correr resultados con él**; los cambios van en `_v2` y el CSV de resultados registra la versión usada.

| Archivo | Agente / uso | Placeholders que rellena el código |
|---|---|---|
| `navegante_v1.txt` | Navegante (propone un movimiento) | `{mapa}` `{pos_f}` `{pos_c}` `{meta_f}` `{meta_c}` `{direccion_meta}` `{historial}` `{vecinos}` `{bloque_correccion}` |
| `navegante_correccion_v1.txt` | Fragmento que se inserta en `{bloque_correccion}` solo en un reintento (vacío en el primer intento) | `{motivo}` `{hechos_libres}` |
| `critico_v1.txt` | Crítico-Corrector (valida, corrige o rechaza un candidato) | `{hechos}` |
| `arbitro_v1.txt` | Árbitro (elige entre candidatos ya válidos) | `{pos_f}` `{pos_c}` `{meta_f}` `{meta_c}` `{candidatos}` |

Se cargan con `Path(...).read_text(encoding="utf-8").format(**campos)`. Los prompts no contienen otras llaves `{}`.

## Formato de lo que genera el código

**`{vecinos}`** (herramienta `observar()`, igual que en el D2), una línea por movimiento, en el orden Arriba, Abajo, Izquierda, Derecha:
```
- Arriba: (3, 4) -> valor 1 [muro]
- Abajo: (5, 4) -> valor 0 [libre]
- Izquierda: (4, 3) -> valor 0 [libre] (ya visitada)
- Derecha: fuera del mapa
```

**`{hechos}`** (herramienta `validar()`), seis líneas siempre en este orden:
```
Posición: (f, c)
Propuesta del Navegante: NOMBRE | (pf, pc) | valor declarado: V
Celda que corresponde al nombre "NOMBRE": (f, c), valor V, libre|muro   —o—   fuera del mapa
Nombre y coordenada coinciden: sí   —o—   no (la coordenada (pf, pc) corresponde a "OTRO"|es la posición actual, no un vecino|no es un vecino)
Coordenada propuesta: valor real V, adyacente: sí   —o—   no es adyacente   —o—   fuera del mapa
Celdas libres vecinas: Arriba (f, c), Izquierda (f, c), ...
```

**`{candidatos}`** (el orquestador, después de deduplicar y contar votos):
```
1. Derecha (4, 5): no visitada | distancia a la meta: 2 | salidas: 2 | votos: 2
2. Abajo (5, 4): visitada | distancia a la meta: 4 | salidas: 1 | votos: 1
```
`distancia a la meta` = distancia Manhattan desde el candidato; `salidas` = celdas libres vecinas del candidato sin contar la celda actual.

## Formatos de salida y cómo parsearlos

| Agente | Formato | Regex sugerido |
|---|---|---|
| Navegante | `Meta hacia: D \| Elijo: MOV \| (f, c) \| valor: V \| adyacente: sí` | el mismo parser del D2 (`validar_propuesta`) |
| Crítico | `Veredicto: APRUEBO` / `Veredicto: CORRIJO \| Movimiento: MOV \| (f, c)` / `Veredicto: RECHAZO \| Motivo: ...` | `Veredicto:\s*(APRUEBO\|CORRIJO\|RECHAZO)` |
| Árbitro | `ELIJO: N \| MOV \| (f, c) \| motivo: ...` | `ELIJO:\s*(\d+)` — usar la celda de la lista de candidatos, no la que reescriba el modelo |

## Decisiones de diseño que conviene declarar en el PDF

- **Few-shot positivo** (5 ejemplos, entre ellos el borde de la columna 8 y el callejón sin salida). Los ejemplos usan posiciones inventadas, no los 20 laberintos de evaluación.
- **"Copia la coordenada de la línea"**: el Navegante no calcula la coordenada, la copia de `{vecinos}`. Ataca el fallo dominante del D2 (nombre ≠ coordenada).
- **Aislamiento de contexto**: el reintento del Navegante recibe el hecho corregido, **no** su propia salida errónea, para evitar que el error se repita en el contexto.
- **Regla de prioridad del Crítico**: si la coordenada es válida, manda la coordenada; si no, manda el nombre. Es una hipótesis: revisar en la prueba de humo con cuál de las dos acierta más el modelo.
- **El Árbitro aplica criterios explícitos.** Para saber si el LLM aporta algo frente a una regla fija, conviene correr una ablación extra `arbitro_regla` (mismos criterios en código).

## Estado

Estos prompts aún **no se probaron con el modelo** (Qwen2.5-1.5B-Instruct). Lo que sí se comprobó:
los placeholders rendericen y que los textos que genera el código (`observar()`, `validar()`, candidatos del Árbitro) sean **idénticos** a los de los ejemplos (`tests/test_formatos.py`); y que las respuestas de los ejemplos sigan las reglas del propio prompt. Ajustar en la prueba de humo de 3 laberintos y, si cambia algo, crear `_v2`.
