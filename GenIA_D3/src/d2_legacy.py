"""Baseline (direct prompting) y agente paso a paso del Entregable 2, copiados SIN cambios del
notebook Entregable2_GenAI.ipynb (celdas 10 y 12). Se usan para volver a correr el baseline y la
solucion del D2 con el mismo codigo si se necesita. Reciben `generar(prompt, max_new_tokens)`."""
import re, time
from .maze import evaluar_ruta, DIRS, bfs

# ---------- Baseline: parser de la salida libre ----------
PATRON = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")
PALABRAS_IGNORAR = ("adyac", "desde", "anterior", "previa")

def parsear_ruta_libre(texto):
    ruta = []
    for linea in texto.splitlines():
        if any(p in linea.lower() for p in PALABRAS_IGNORAR):
            continue
        m = PATRON.search(linea)
        if m:
            p = (int(m.group(1)), int(m.group(2)))
            if not ruta or ruta[-1] != p:
                ruta.append(p)
    return ruta

def prompt_baseline(lab, inicio, meta):
    filas, columnas = len(lab), len(lab[0])
    matriz_str = "\n".join(str(f) for f in lab)
    return f"""
Dada la siguiente matriz de un laberinto de TAMAÑO ESPECÍFICO {filas} filas por {columnas} columnas:

{matriz_str}

RESTRICCIONES DE DIMENSIÓN Y MAPA:
- El mapa tiene exactamente {filas} filas (índices 0 a {filas-1}) y {columnas} columnas (índices 0 a {columnas-1}).
- '0' es un camino libre y '1' es una muralla infranqueable.
- Inicio: ({inicio[0]}, {inicio[1]}) [Fila {inicio[0]}, Columna {inicio[1]}]
- Meta: ({meta[0]}, {meta[1]}) [Fila {meta[0]}, Columna {meta[1]}]

REGLA ESTRICTA DE DIAGNÓSTICO:
Propón una ruta para llegar a la meta. Para CADA PASO que escribas, debes incluir obligatoriamente:
1. La coordenada propuesta: (Fila, Columna).
2. El valor real en la matriz para esa coordenada (¿Es 0 o 1?).
3. Verificación de adyacencia: Confirma que el movimiento es exactamente a 1 casilla de distancia adyacente (Arriba, Abajo, Izquierda o Derecha) desde la coordenada anterior.

Escribe la solución paso a paso.
"""

def correr_baseline(generar, lab, inicio, meta, max_new_tokens=1000):
    t0 = time.time()
    salida = generar(prompt_baseline(lab, inicio, meta), max_new_tokens)
    ruta = parsear_ruta_libre(salida)
    estado, ruta_eval = evaluar_ruta(lab, inicio, meta, ruta)
    return {"estado": estado, "ruta": ruta_eval, "pasos": max(len(ruta_eval)-1, 0),
            "invalidas": None, "valor_mal_declarado": None, "coordenada_mal_declarada": None,
            "tiempo_s": round(time.time()-t0, 1), "salida_cruda": salida}


# ---------- Solución: agente paso a paso ----------
# DIRS viene de maze.py (misma definicion del D2)
LETRAS = "ABCD"
PATRON_VALOR = re.compile(r"(?:valor\s*:?\s*|\)\s*\|\s*)(\d)\b", re.IGNORECASE)
PATRON_LETRA = re.compile(r"\W*(?:[Oo]pci[oó]n\s*)?([ABCD])\b")

def opciones(lab, pos):
    """Las 4 opciones de movimiento: (letra, nombre, celda o None si está fuera del mapa)."""
    n = len(lab); ops = []
    for letra, (nombre, dr, dc) in zip(LETRAS, DIRS):
        r, c = pos[0]+dr, pos[1]+dc
        ops.append((letra, nombre, (r, c) if 0 <= r < n and 0 <= c < n else None))
    return ops

def direccion_meta(pos, meta):
    partes = []
    df, dc = meta[0]-pos[0], meta[1]-pos[1]
    if df: partes.append(f"{'Abajo' if df > 0 else 'Arriba'} ({abs(df)} filas)")
    if dc: partes.append(f"{'Derecha' if dc > 0 else 'Izquierda'} ({abs(dc)} columnas)")
    return " y ".join(partes)

NOMBRE_A_LETRA = {"arriba": "A", "abajo": "B", "izquierda": "C", "derecha": "D"}
LETRA_A_NOMBRE = {"A": "Arriba", "B": "Abajo", "C": "Izquierda", "D": "Derecha"}
PATRON_NOMBRE = re.compile(r"\W*(arriba|abajo|izquierda|derecha)\b", re.IGNORECASE)

def prompt_paso(lab, pos, meta, visitadas, historial, error=None, usar_memoria=True, rechazadas=(),
                usar_direccion=True):
    n = len(lab)
    matriz_str = "\n".join(str(f) for f in lab)
    lineas = []
    for letra, nombre, celda in opciones(lab, pos):
        if celda is None:
            linea = f"- {nombre}: fuera del mapa"
        else:
            v = lab[celda[0]][celda[1]]
            linea = f"- {nombre}: ({celda[0]}, {celda[1]}) -> valor {v} [{'muro' if v == 1 else 'libre'}]"
            if usar_memoria and v == 0 and celda in visitadas:
                linea += " (ya visitada)"
        if letra in rechazadas:
            linea += "  <- RECHAZADA"
        lineas.append(linea)
    texto = f"""Estás navegando un laberinto de {n}x{n} (0 = libre, 1 = muro infranqueable).

Mapa:
{matriz_str}

Estás en ({pos[0]}, {pos[1]}). La meta es ({meta[0]}, {meta[1]}).
"""
    if usar_direccion:
        texto += f"La meta está hacia: {direccion_meta(pos, meta)}.\n"
    if usar_memoria:
        texto += "Últimos pasos: " + " -> ".join(f"({r}, {c})" for r, c in historial[-8:]) + "\n"
    texto += "\nMovimientos posibles:\n" + "\n".join(lineas) + "\n\n"
    texto += "Elige un movimiento hacia una celda libre que te acerque a la meta. No elijas muros ni movimientos fuera del mapa.\n"
    if usar_memoria:
        texto += "Prefiere celdas no visitadas. Si todas las celdas libres ya fueron visitadas, estás en un callejón sin salida: retrocede a una celda ya visitada.\n"
    if error:
        texto += f"\nTu propuesta anterior fue RECHAZADA: {error}. Elige otro movimiento.\n"
    texto += """
Responde en UNA sola línea con este formato:
Meta hacia: DIRECCIÓN | Elijo: MOVIMIENTO | (fila, columna) | valor: V | adyacente: sí
donde MOVIMIENTO es Arriba, Abajo, Izquierda o Derecha."""
    return texto

def validar_propuesta(lab, pos, texto):
    """Devuelve (celda, error, valor_declarado, coordenada_mal_declarada, letra).
    La decisión es el movimiento escrito después de "Elijo" (palabra o letra); si no hay,
    se usa la primera coordenada distinta de la posición actual."""
    n = len(lab)
    partes = re.split(r"[Ee]lijo", texto)
    if len(partes) > 1:
        texto = partes[-1]
    ops = {l: celda for l, _, celda in opciones(lab, pos)}
    coords = [(int(a), int(b)) for a, b in PATRON.findall(texto)]
    mv = PATRON_VALOR.search(texto)
    valor_decl = int(mv.group(1)) if mv else None
    mn = PATRON_NOMBRE.match(texto)
    ml = PATRON_LETRA.match(texto)
    letra = NOMBRE_A_LETRA[mn.group(1).lower()] if mn else (ml.group(1) if ml else None)
    if letra:
        celda = ops[letra]
        if celda is None:
            return None, f"{LETRA_A_NOMBRE[letra]} está fuera del mapa", valor_decl, False, letra
        coord_mal = bool(coords) and coords[0] != celda
    else:
        candidatas = [p for p in coords if p != pos]
        if not candidatas:
            return None, "no indicaste un movimiento (Arriba, Abajo, Izquierda o Derecha)", valor_decl, False, None
        celda = candidatas[0]; coord_mal = False
        if not (0 <= celda[0] < n and 0 <= celda[1] < n):
            return None, f"({celda[0]}, {celda[1]}) está fuera del mapa", valor_decl, False, None
        if abs(celda[0]-pos[0]) + abs(celda[1]-pos[1]) != 1:
            return None, f"({celda[0]}, {celda[1]}) no es adyacente a ({pos[0]}, {pos[1]})", valor_decl, False, None
        letra = next((l for l, c in ops.items() if c == celda), None)
    if lab[celda[0]][celda[1]] == 1:
        return None, f"{LETRA_A_NOMBRE.get(letra, '')} ({celda[0]}, {celda[1]}) es un muro (valor 1)".strip(), valor_decl, coord_mal, letra
    return celda, None, valor_decl, coord_mal, letra

def correr_agente(generar, lab, inicio, meta, max_pasos=81, max_reintentos=3,
                  max_new_tokens=60, verbose=False, usar_memoria=True, usar_direccion=True):
    t0 = time.time()
    pos = inicio; ruta = [inicio]; visitadas = {inicio}
    invalidas = 0; valor_mal = 0; coord_mal_total = 0; log = []; estado = None
    while pos != meta and len(ruta)-1 < max_pasos:
        error = None; nuevo = None; rechazadas = set()
        for intento in range(max_reintentos + 1):
            salida = generar(prompt_paso(lab, pos, meta, visitadas, ruta, error, usar_memoria, rechazadas, usar_direccion),
                             max_new_tokens)
            nuevo, error, valor_decl, coord_mal, letra = validar_propuesta(lab, pos, salida)
            log.append({"pos": pos, "intento": intento, "salida": salida.strip(), "error": error})
            if verbose:
                marca = "OK " if error is None else "RECHAZADO"
                print(f"  {marca} en {pos}: {salida.strip()[:140]}" + (f"  -> {error}" if error else ""))
            coord_mal_total += int(coord_mal)
            if error is None:
                if valor_decl is not None and valor_decl != lab[nuevo[0]][nuevo[1]]:
                    valor_mal += 1
                break
            invalidas += 1
            if letra:
                rechazadas.add(letra)
        if nuevo is None:
            estado = "propuestas_invalidas"
            break
        pos = nuevo; ruta.append(pos); visitadas.add(pos)
    if estado is None:
        estado, ruta = evaluar_ruta(lab, inicio, meta, ruta)
        if estado == "no_llega_meta":
            estado = "limite_pasos"
    return {"estado": estado, "ruta": ruta, "pasos": len(ruta)-1, "invalidas": invalidas,
            "valor_mal_declarado": valor_mal, "coordenada_mal_declarada": coord_mal_total,
            "tiempo_s": round(time.time()-t0, 1), "salida_cruda": log}

