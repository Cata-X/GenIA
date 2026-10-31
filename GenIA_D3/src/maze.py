"""Laberintos, verificador y utilidades del Entregable 2, copiados SIN cambios del notebook
Entregable2_GenAI.ipynb (celdas 6, 8 y 14) para que el D3 use exactamente el mismo criterio de
correccion y los mismos 20 laberintos. Solo se agregaron los imports y la constante DIRS."""
import random, re, json, time
from collections import deque

DIRS = [("Arriba", -1, 0), ("Abajo", 1, 0), ("Izquierda", 0, -1), ("Derecha", 0, 1)]

import random, re, json, time
from collections import deque

# ---------- Laberintos ----------
LAB_D1 = [
    [0,0,1,0,0,0,0,0,0],
    [1,0,1,0,1,1,1,1,0],
    [0,0,0,0,1,0,0,1,0],
    [0,1,1,1,1,0,1,1,0],
    [0,0,0,0,0,0,0,1,0],
    [1,1,1,1,1,1,0,1,0],
    [0,0,0,0,0,1,0,0,0],
    [0,1,1,1,0,1,1,1,1],
    [0,0,0,0,0,0,0,0,0],
]

def bfs(lab, inicio, meta):
    n = len(lab)
    prev = {inicio: None}
    q = deque([inicio])
    while q:
        r, c = q.popleft()
        if (r, c) == meta:
            camino = []
            p = meta
            while p is not None:
                camino.append(p); p = prev[p]
            return camino[::-1]
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < n and 0 <= nc < n and lab[nr][nc] == 0 and (nr,nc) not in prev:
                prev[(nr,nc)] = (r,c); q.append((nr,nc))
    return None

def generar_laberinto(seed, n=9, muros_extra=3, dist_min=10):
    """Laberinto perfecto (backtracker) en 9x9 + algunos muros abiertos para crear ciclos."""
    rng = random.Random(seed)
    lab = [[1]*n for _ in range(n)]
    k = (n+1)//2
    visit = {(0,0)}; pila = [(0,0)]; lab[0][0] = 0
    while pila:
        i, j = pila[-1]
        vec = [(i+di, j+dj) for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]
               if 0 <= i+di < k and 0 <= j+dj < k and (i+di, j+dj) not in visit]
        if not vec:
            pila.pop(); continue
        ni, nj = rng.choice(vec)
        lab[i+ni][j+nj] = 0          # celda muro entre ambas
        lab[2*ni][2*nj] = 0
        visit.add((ni, nj)); pila.append((ni, nj))
    # abrir algunos muros internos para que haya más de un camino
    candidatos = [(r,c) for r in range(n) for c in range(n)
                  if lab[r][c] == 1 and ((r%2==0 and c%2==1) or (r%2==1 and c%2==0))]
    for r, c in rng.sample(candidatos, min(muros_extra, len(candidatos))):
        lab[r][c] = 0
    libres = [(r,c) for r in range(n) for c in range(n) if lab[r][c] == 0]
    while True:
        a, b = rng.sample(libres, 2)
        camino = bfs(lab, a, b)
        if camino and len(camino)-1 >= dist_min:
            return lab, a, b

def construir_dataset(N, seed_base=2026):
    data = [{"id": 0, "laberinto": LAB_D1, "inicio": (0,0), "meta": (0,8), "origen": "Entregable 1"}]
    for i in range(1, N):
        lab, a, b = generar_laberinto(seed_base + i)
        data.append({"id": i, "laberinto": lab, "inicio": a, "meta": b, "origen": f"seed {seed_base+i}"})
    return data


# ---------- Verificador (mismo criterio para todos los métodos) ----------
def evaluar_ruta(lab, inicio, meta, ruta):
    n = len(lab)
    if not ruta:
        return "sin_ruta", []
    ruta = list(ruta)
    if ruta[0] != inicio:
        ruta = [inicio] + ruta
    for idx, (r, c) in enumerate(ruta):
        if not (0 <= r < n and 0 <= c < n):
            return "fuera_de_rango", ruta[:idx+1]
        if lab[r][c] == 1:
            return "pisa_muro", ruta[:idx+1]
        if idx > 0:
            pr, pc = ruta[idx-1]
            if abs(r-pr) + abs(c-pc) != 1:
                return "salto_no_adyacente", ruta[:idx+1]
        if (r, c) == meta:
            return "exito", ruta[:idx+1]
    return "no_llega_meta", ruta


# ---------- Control: política aleatoria en el mismo entorno ----------
def correr_aleatorio(lab, inicio, meta, max_pasos=81, seed=0):
    rng = random.Random(seed); n = len(lab)
    pos = inicio; ruta = [inicio]
    while pos != meta and len(ruta)-1 < max_pasos:
        opciones = [(pos[0]+dr, pos[1]+dc) for _, dr, dc in DIRS
                    if 0 <= pos[0]+dr < n and 0 <= pos[1]+dc < n and lab[pos[0]+dr][pos[1]+dc] == 0]
        pos = rng.choice(opciones); ruta.append(pos)
    estado, ruta = evaluar_ruta(lab, inicio, meta, ruta)
    if estado == "no_llega_meta":
        estado = "limite_pasos"
    return {"estado": estado, "ruta": ruta, "pasos": len(ruta)-1, "invalidas": 0,
            "valor_mal_declarado": 0, "coordenada_mal_declarada": 0, "tiempo_s": 0.0, "salida_cruda": ""}

def dibujar(lab, ruta, inicio, meta):
    en_ruta = set(ruta)
    for r, fila in enumerate(lab):
        s = ""
        for c, v in enumerate(fila):
            if (r, c) == inicio: s += " I"
            elif (r, c) == meta: s += " M"
            elif v == 1: s += " █"
            elif (r, c) in en_ruta: s += " ·"
            else: s += "  "
        print(s)

def analizar_fallo(lab, ruta):
    """Datos para explicar POR QUÉ falla: revisitas y callejones sin salida."""
    from collections import Counter
    n = len(lab)
    cuenta = Counter(ruta)
    revisitadas = [(p, k) for p, k in cuenta.most_common() if k > 1]
    def libres(p):
        return sum(1 for _, dr, dc in DIRS
                   if 0 <= p[0]+dr < n and 0 <= p[1]+dc < n and lab[p[0]+dr][p[1]+dc] == 0)
    callejones = sorted({p for p in ruta if libres(p) == 1})
    print(f"Pasos totales: {len(ruta)-1} | celdas distintas: {len(cuenta)}")
    print("Celdas más revisitadas (celda, veces):", revisitadas[:6])
    print("Callejones sin salida que visitó:", callejones)

