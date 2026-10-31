"""Los tres agentes del Entregable 3 y los parsers de sus salidas.

Un agente aqui = una llamada al MISMO modelo (Qwen2.5-1.5B-Instruct) con su rol, su prompt
versionado, una entrada y una salida definidas, y una decision que cambia lo que pasa despues:

    Navegante  entrada: situacion (observar())      salida: propuesta de movimiento
    Critico    entrada: hechos (validar())          salida: APRUEBO | CORRIJO | RECHAZO
    Arbitro    entrada: candidatos ya validos       salida: ELIJO n

Los parsers son tolerantes (toman la primera linea que contiene la marca) y devuelven None en los
campos que no pudieron leer; el orquestador decide que hacer con eso.
"""
import re
from dataclasses import dataclass

from .prompts import render_navegante, render_critico, render_arbitro
from .tools import NOMBRES

_CANON = {n.lower(): n for n in NOMBRES}
PAT_COORD = re.compile(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")   # admite negativos: (-1, 5) = fuera del mapa
PAT_NOMBRE = re.compile(r"\b(arriba|abajo|izquierda|derecha)\b", re.I)
PAT_VALOR = re.compile(r"valor\s*:?\s*(\d)", re.I)


def _linea(texto, marca):
    """Primera linea que contiene `marca` (regex, sin distinguir mayusculas); si no hay, la primera no vacia."""
    lineas = [l for l in texto.splitlines() if l.strip()]
    for l in lineas:
        if re.search(marca, l, re.I):
            return l
    return lineas[0] if lineas else ""


# ---------------------------------------------------------------------------- Navegante
@dataclass
class Propuesta:
    crudo: str
    nombre: str | None
    coord: tuple | None
    valor: int | None

    @property
    def clave(self):
        return (self.nombre, self.coord, self.valor)


def parse_navegante(texto):
    """`Meta hacia: D | Elijo: MOV | (f, c) | valor: V | adyacente: sí`.
    Se descarta el segmento "Meta hacia" para no confundir la direccion de la meta con la decision."""
    linea = _linea(texto, r"elijo")
    segs = [s.strip() for s in linea.split("|")]
    segs = [s for s in segs if not re.match(r"meta\s+hacia", s, re.I)]
    cuerpo = " | ".join(segs)
    cuerpo = re.sub(r"^\s*elijo\s*:?\s*", "", cuerpo, flags=re.I)
    mn = PAT_NOMBRE.search(cuerpo)
    mc = PAT_COORD.search(cuerpo)
    mv = PAT_VALOR.search(cuerpo)
    return Propuesta(
        crudo=texto.strip(),
        nombre=_CANON[mn.group(1).lower()] if mn else None,
        coord=(int(mc.group(1)), int(mc.group(2))) if mc else None,
        valor=int(mv.group(1)) if mv else None)


class Navegante:
    nombre = "Navegante"

    def __init__(self, llm, plantilla, version="v1", observar=True, max_new_tokens=60):
        self.llm, self.plantilla, self.version = llm, plantilla, version
        self.observar, self.max_new_tokens = observar, max_new_tokens

    def prompt(self, obs, entorno, correccion=""):
        return render_navegante(self.plantilla, obs, entorno, correccion, self.version, self.observar)

    def proponer(self, prompt, k, temperatura, seed):
        """k propuestas. k=1 -> greedy (temperatura ignorada); k>1 -> muestreo con `temperatura`."""
        salidas = self.llm.generar(prompt, self.max_new_tokens, n=k, temperatura=temperatura if k > 1 else 0.0, seed=seed)
        return [parse_navegante(s) for s in salidas]


# ---------------------------------------------------------------------------- Critico
@dataclass
class Veredicto:
    crudo: str
    veredicto: str | None      # APRUEBO | CORRIJO | RECHAZO | None (no se pudo leer)
    nombre: str | None = None
    coord: tuple | None = None
    motivo: str | None = None


def parse_critico(texto):
    linea = _linea(texto, r"veredicto")
    m = re.search(r"veredicto\s*:?\s*(APRUEBO|CORRIJO|RECHAZO)", linea, re.I)
    if not m:
        return Veredicto(crudo=texto.strip(), veredicto=None)
    v = m.group(1).upper()
    nombre = coord = motivo = None
    if v == "CORRIJO":
        resto = linea[m.end():]
        mn, mc = PAT_NOMBRE.search(resto), PAT_COORD.search(resto)
        nombre = _CANON[mn.group(1).lower()] if mn else None
        coord = (int(mc.group(1)), int(mc.group(2))) if mc else None
    elif v == "RECHAZO":
        mm = re.search(r"motivo\s*:\s*(.+)$", linea, re.I)
        motivo = mm.group(1).strip() if mm else None
    return Veredicto(crudo=texto.strip(), veredicto=v, nombre=nombre, coord=coord, motivo=motivo)


class Critico:
    nombre = "Crítico"

    def __init__(self, llm, version="v1", max_new_tokens=80):
        self.llm, self.version, self.max_new_tokens = llm, version, max_new_tokens

    def prompt(self, hechos_texto):
        return render_critico(hechos_texto, self.version)

    def revisar_lote(self, prompts):
        """Revisa varios prompts distintos (greedy). Devuelve una lista de Veredicto."""
        salidas = self.llm.generar_lote(prompts, self.max_new_tokens)
        return [parse_critico(s) for s in salidas]


# ---------------------------------------------------------------------------- Arbitro
@dataclass
class Eleccion:
    crudo: str
    numero: int | None


def parse_arbitro(texto):
    linea = _linea(texto, r"elijo")
    m = re.search(r"elijo\s*:?\s*(\d+)", linea, re.I)
    return Eleccion(crudo=texto.strip(), numero=int(m.group(1)) if m else None)


class Arbitro:
    nombre = "Árbitro"

    def __init__(self, llm, version="v1", max_new_tokens=40):
        self.llm, self.version, self.max_new_tokens = llm, version, max_new_tokens

    def prompt(self, entorno, candidatos_txt):
        return render_arbitro(entorno, candidatos_txt, self.version)

    def elegir(self, prompt):
        return parse_arbitro(self.llm.generar(prompt, self.max_new_tokens, n=1, temperatura=0.0, seed=None)[0])


def clave_regla(c, meta):
    """Orden de los criterios del prompt del Arbitro (menor = mejor). Se usa en la ablacion
    `arbitro='regla'` y como referencia en los tests."""
    return (0 if c["celda"] == tuple(meta) else 1,
            1 if c["salidas"] == 0 else 0,
            1 if c["visitada"] else 0,
            c["distancia"],
            -c["votos"])
