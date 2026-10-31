"""LLM simulado (SIN modelo): sirve para probar el pipeline completo en un computador sin GPU.

NO es el modelo del trabajo y sus resultados no se reportan. Lee los prompts reales (los mismos
archivos prompts/*.txt, ya rellenados) como lo haria un modelo, y responde en el formato pedido,
con errores inyectados con cierta probabilidad:

    Navegante: con prob. p_coord_mal escribe el nombre de un movimiento y la coordenada de OTRO
               (el fallo dominante del D2); con prob. p_muro elige una celda que no es libre.
               En un reintento (bloque <correccion>) responde bien.
    Critico:   aplica las reglas del prompt a los hechos; con prob. p_critico_err aprueba sin mirar.
    Arbitro:   aplica los criterios del prompt; con prob. p_arbitro_err elige el candidato 1.

Mismo contrato que `QwenLLM`: generar(...) y generar_lote(...).
"""
import json
import random
import re

from .tools import NOMBRES, DELTA

_NAV = re.compile(r"Estás en \((\d+), (\d+)\)\. La meta es \((\d+), (\d+)\)")
_COORD = re.compile(r"\((-?\d+), (-?\d+)\)")


class LLMSimulado:
    def __init__(self, p_coord_mal=0.30, p_muro=0.10, p_critico_err=0.05, p_arbitro_err=0.10, seed=0):
        self.p_coord_mal, self.p_muro = p_coord_mal, p_muro
        self.p_critico_err, self.p_arbitro_err = p_critico_err, p_arbitro_err
        self.rng = random.Random(seed)
        self.llamadas = 0
        self.segundos = 0.0

    # ------------------------------------------------------------ interfaz
    def generar(self, prompt, max_new_tokens=60, n=1, temperatura=0.0, seed=None):
        self.llamadas += 1
        rng = random.Random(seed) if seed is not None else self.rng
        n = n if temperatura else 1
        return [self._responder(prompt, rng) for _ in range(n)]

    def generar_lote(self, prompts, max_new_tokens=60):
        self.llamadas += 1
        return [self._responder(p, self.rng) for p in prompts]

    def como_funcion_d2(self):
        """Para el baseline / agente del D2 en ensayos en seco: devuelve texto sin sentido."""
        return lambda prompt, max_new_tokens: "(0, 0)"

    # ------------------------------------------------------------ despacho
    def _responder(self, prompt, rng):
        cabeza = prompt.lstrip()[:60]
        if "Eres el Navegante" in cabeza:
            return self._navegante(prompt, rng)
        if "Eres el Crítico" in cabeza:
            return self._critico(prompt, rng)
        if "Eres el Árbitro" in cabeza:
            return self._arbitro(prompt, rng)
        return "no entiendo"

    # ------------------------------------------------------------ Navegante
    def _navegante(self, prompt, rng):
        sit = prompt.split("<situacion>")[-1]
        mapa = [json.loads(l) for l in re.findall(r"^\[[01, ]+\]$", sit, re.M)]
        f, c, mf, mc = map(int, _NAV.findall(sit)[-1])
        hist = [tuple(map(int, x)) for x in _COORD.findall(re.findall(r"Últimos pasos: (.*)", sit)[-1])]
        n = len(mapa)
        reintento = "<correccion>" in sit

        def celda(nombre):
            dr, dc = DELTA[nombre]
            r, k = f + dr, c + dc
            return (r, k) if 0 <= r < n and 0 <= k < n else None

        libres = [(nm, celda(nm)) for nm in NOMBRES if celda(nm) and mapa[celda(nm)[0]][celda(nm)[1]] == 0]
        bueno = min(libres, key=lambda x: ((x[1] in hist), abs(x[1][0] - mf) + abs(x[1][1] - mc), rng.random()))
        nombre, cell = bueno
        if not reintento:
            x = rng.random()
            if x < self.p_muro:
                muros = [nm for nm in NOMBRES if celda(nm) is None or mapa[celda(nm)[0]][celda(nm)[1]] == 1]
                if muros:
                    nombre = rng.choice(muros)
                    cell = celda(nombre) or (f + DELTA[nombre][0], c + DELTA[nombre][1])
            elif x < self.p_muro + self.p_coord_mal:
                otros = [nm for nm in NOMBRES if nm != nombre]
                otro = rng.choice(otros)
                cell_otro = celda(otro) or (f + DELTA[otro][0], c + DELTA[otro][1])
                cell = cell_otro                           # nombre != coordenada
        val = mapa[cell[0]][cell[1]] if 0 <= cell[0] < n and 0 <= cell[1] < n else 0
        return f"Meta hacia: Abajo | Elijo: {nombre} | ({cell[0]}, {cell[1]}) | valor: {val} | adyacente: sí"

    # ------------------------------------------------------------ Critico
    def _critico(self, prompt, rng):
        h = re.findall(r"<hechos>\n(Posición:.*?)\n</hechos>", prompt, re.S)[-1]
        pos = tuple(map(int, _COORD.search(re.search(r"Posición: .*", h).group(0)).groups()))
        l_prop = re.search(r"Propuesta del Navegante: (\w+) \| (.*?) \|", h)
        nombre, coord_txt = l_prop.group(1), l_prop.group(2)
        mc = _COORD.search(coord_txt)
        coord = tuple(map(int, mc.groups())) if mc else None
        libres = {nm: tuple(map(int, xy)) for nm, *xy in re.findall(r"(\w+) \((\d+), (\d+)\)", re.search(r"Celdas libres vecinas: (.*)", h).group(1))}
        celda_nombre = re.search(r'Celda que corresponde al nombre "\w+": (.*)', h).group(1)
        nombre_libre = "libre" in celda_nombre
        coincide = "coinciden: sí" in h
        coord_valida = "adyacente: sí" in h and "valor real 0" in h
        if rng.random() < self.p_critico_err:
            return "Veredicto: APRUEBO"
        if coincide and nombre_libre:
            return "Veredicto: APRUEBO"
        if coord_valida:
            nm = next(k for k, v in libres.items() if v == coord)
            return f"Veredicto: CORRIJO | Movimiento: {nm} | ({coord[0]}, {coord[1]})"
        if nombre_libre:
            v = libres[nombre]
            return f"Veredicto: CORRIJO | Movimiento: {nombre} | ({v[0]}, {v[1]})"
        return f"Veredicto: RECHAZO | Motivo: {celda_nombre}. Celdas libres: " + ", ".join(
            f"{k} ({v[0]}, {v[1]})" for k, v in libres.items())

    # ------------------------------------------------------------ Arbitro
    def _arbitro(self, prompt, rng):
        sit = prompt.split("<situacion>")[-1]
        mf, mc = map(int, _NAV.findall(sit)[-1][2:])
        cands = []
        for num, nm, r, c, vis, dist, sal, vot in re.findall(
                r"(\d+)\. (\w+) \((\d+), (\d+)\): (no visitada|visitada) \| distancia a la meta: (\d+) \| salidas: (\d+) \| votos: (\d+)", sit):
            cands.append((int(num), nm, (int(r), int(c)), vis == "visitada", int(dist), int(sal), int(vot)))
        if rng.random() < self.p_arbitro_err:
            best = cands[0]
        else:
            best = min(cands, key=lambda x: (0 if x[2] == (mf, mc) else 1, 1 if x[5] == 0 else 0,
                                             1 if x[3] else 0, x[4], -x[6], x[0]))
        return f"ELIJO: {best[0]} | {best[1]} | ({best[2][0]}, {best[2][1]}) | motivo: simulado"
