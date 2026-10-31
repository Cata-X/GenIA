"""Los textos que genera el codigo (tools.py) deben ser IDENTICOS a los ejemplos de los prompts.
Si el formato que ve el modelo en produccion difiere del de los ejemplos, el few-shot deja de servir."""
import re
from pathlib import Path

import pytest

from src.tools import Entorno, NOMBRES, DELTA
from src.agents import clave_regla
from src.prompts import DIR_PROMPTS

DIRS = DELTA


def ejemplos(archivo):
    t = (DIR_PROMPTS / archivo).read_text(encoding="utf-8")
    return re.findall(r"<ejemplo>(.*?)</ejemplo>", t.split("<ejemplos>")[1].split("</ejemplos>")[0], re.S)


def lab_vacio(n=9):
    return [[1] * n for _ in range(n)]


# ------------------------------------------------------------------------- Navegante: observar()
@pytest.mark.parametrize("i,ex", list(enumerate(ejemplos("navegante_v1.txt"), 1)))
def test_observar_reproduce_ejemplo_navegante(i, ex):
    pr, pc, mr, mc = map(int, re.search(r"Estás en \((\d+), (\d+)\)\. La meta es \((\d+), (\d+)\)", ex).groups())
    lineas = re.findall(r"^- (?:Arriba|Abajo|Izquierda|Derecha): .*$", ex, re.M)
    lab, visitadas = lab_vacio(), set()
    for l in lineas:
        nombre = l[2:].split(":")[0]
        dr, dc = DIRS[nombre]
        r, c = pr + dr, pc + dc
        if "fuera del mapa" in l:
            continue
        lab[r][c] = 0 if "[libre]" in l else 1
        if "(ya visitada)" in l:
            visitadas.add((r, c))
    env = Entorno(lab, (pr, pc), (mr, mc))
    env.visitadas |= visitadas
    obs = env.observar()
    assert obs.vecinos.splitlines() == lineas
    esperado_dir = re.search(r"La meta está hacia: (.*)\.", ex).group(1)
    assert obs.direccion_meta == esperado_dir


# ------------------------------------------------------------------------- Critico: validar()
@pytest.mark.parametrize("i,ex", list(enumerate(ejemplos("critico_v1.txt"), 1)))
def test_validar_reproduce_ejemplo_critico(i, ex):
    hechos = re.search(r"<hechos>\n(.*?)\n</hechos>", ex, re.S).group(1)
    pos = tuple(map(int, re.search(r"Posición: \((\d+), (\d+)\)", hechos).groups()))
    nombre, cf, cc, valor = re.search(r"Propuesta del Navegante: (\w+) \| \((\d+), (\d+)\) \| valor declarado: (\d)", hechos).groups()
    lab = lab_vacio()
    libres = re.search(r"Celdas libres vecinas: (.*)", hechos).group(1)
    for r, c in re.findall(r"\((\d+), (\d+)\)", libres):
        lab[int(r)][int(c)] = 0
    env = Entorno(lab, pos, (8, 8))
    h = env.validar(nombre, (int(cf), int(cc)), int(valor))
    assert h.texto == hechos
    veredicto = re.search(r"Veredicto: (APRUEBO|CORRIJO|RECHAZO)", ex).group(1)
    assert h.esperado[0] == veredicto
    if veredicto == "CORRIJO":
        mm = re.search(r"Movimiento: (\w+) \| \((\d+), (\d+)\)", ex)
        assert h.esperado[1] == mm.group(1) and h.esperado[2] == (int(mm.group(2)), int(mm.group(3)))


def test_validar_casos_borde():
    lab = lab_vacio(); lab[4][5] = 0; lab[4][3] = 0
    env = Entorno(lab, (4, 4), (8, 8))
    # sin coordenada
    h = env.validar("Derecha", None, None)
    assert "(sin coordenada)" in h.texto and "valor declarado: no indicado" in h.texto
    assert h.esperado == ("CORRIJO", "Derecha", (4, 5))
    # coordenada fuera del mapa (negativa) y nombre fuera del mapa
    env2 = Entorno(lab_vacio(), (0, 0), (8, 8)); env2.lab[0][1] = 0
    h = env2.validar("Arriba", (-1, 0), 0)
    assert "fuera del mapa" in h.texto and h.esperado[0] == "RECHAZO"
    # coordenada = otro vecino libre, nombre = muro  -> manda la coordenada
    h = env.validar("Arriba", (4, 3), 0)
    assert h.esperado == ("CORRIJO", "Izquierda", (4, 3))


# ------------------------------------------------------------------------- Arbitro: candidatos
@pytest.mark.parametrize("i,ex", list(enumerate(ejemplos("arbitro_v1.txt"), 1)))
def test_candidatos_reproducen_ejemplo_arbitro(i, ex):
    pr, pc, mr, mc = map(int, re.search(r"Estás en \((\d+), (\d+)\)\. La meta es \((\d+), (\d+)\)", ex).groups())
    bloque = re.search(r"Candidatos:\n(.*?)\nRespuesta:", ex, re.S).group(1)
    filas = re.findall(r"(\d+)\. (\w+) \((\d+), (\d+)\): (no visitada|visitada) \| distancia a la meta: (\d+) \| salidas: (\d+) \| votos: (\d+)", bloque)
    # Se talla un mapa 9x9 que produzca exactamente esas estadisticas: cada candidato necesita `salidas`
    # vecinos libres (sin contar la celda actual); se prueban las combinaciones hasta que todas calcen.
    import itertools
    datos = [(nombre, int(r), int(c), vis == "visitada", int(sal), int(vot)) for _, nombre, r, c, vis, _, sal, vot in filas]
    opciones = []
    for _, r, c, _, sal, _ in [(d[0], d[1], d[2], d[3], d[4], d[5]) for d in datos]:
        extra = [(r + dr, c + dc) for dr, dc in DIRS.values()
                 if 0 <= r + dr < 9 and 0 <= c + dc < 9 and (r + dr, c + dc) != (pr, pc)]
        opciones.append(list(itertools.combinations(extra, sal)))
    cands, ok = None, False
    for combo in itertools.product(*opciones):
        lab = lab_vacio(); lab[pr][pc] = 0
        env = Entorno(lab, (pr, pc), (mr, mc))
        for (nombre, r, c, vis, sal, vot), extras in zip(datos, combo):
            lab[r][c] = 0
            for rr, cc in extras:
                lab[rr][cc] = 0
            if vis:
                env.visitadas.add((r, c))
        cands = [env.candidato((d[1], d[2]), d[5]) for d in datos]
        if env.texto_candidatos(cands) == bloque:
            ok = True
            break
    assert ok, "ningun mapa 9x9 reproduce las estadisticas del ejemplo: el ejemplo es inconsistente"
    elegido = int(re.search(r"ELIJO: (\d+)", ex).group(1))
    mejor = min(range(len(cands)), key=lambda k: (clave_regla(cands[k], (mr, mc)), k))
    assert mejor + 1 == elegido, "la regla en codigo no coincide con la respuesta del ejemplo"
