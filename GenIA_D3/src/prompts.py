"""Carga de prompts versionados (prompts/*_vN.txt) y transformaciones para las ablaciones.

Los prompts viven en archivos, fuera del codigo. Aqui solo se leen, se rellenan y, para las
ablaciones, se les quita un bloque. Ninguna ablacion edita los archivos.
"""
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DIR_PROMPTS = RAIZ / "prompts"


def cargar(nombre, version="v1"):
    """Lee prompts/{nombre}_{version}.txt"""
    return (DIR_PROMPTS / f"{nombre}_{version}.txt").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------- Navegante
_RE_EJEMPLOS = re.compile(r"<ejemplos>.*?</ejemplos>\n*", re.S)
_RE_BLOQUE_VECINOS_EJ = re.compile(r"Movimientos posibles:\n(?:- (?:Arriba|Abajo|Izquierda|Derecha): .*\n)+")
_FRASE_COPIA = ('Escribe el nombre del movimiento y COPIA la coordenada de esa misma línea de '
                '"Movimientos posibles"; no la calcules de nuevo.')
_FRASE_COPIA_SIN = ("Escribe el nombre del movimiento y calcula su coordenada con la geometría de arriba; "
                    "comprueba en el mapa que sea una celda libre (valor 0).")
_FRASE_VISITADA = ('Prefiere celdas que no estén marcadas "(ya visitada)". Si todas las celdas libres '
                   'están visitadas, estás en un callejón sin salida: retrocede a una de ellas.')
_FRASE_VISITADA_SIN = ('Prefiere celdas que no aparezcan en "Últimos pasos". Si todas las celdas libres '
                       'aparecen ahí, estás en un callejón sin salida: retrocede.')


def plantilla_navegante(version="v1", fewshot=True, observar=True):
    """Plantilla del Navegante con los bloques de ablacion aplicados.

    fewshot=False  -> se elimina el bloque <ejemplos>.
    observar=False -> se elimina la lista "Movimientos posibles" (del prompt y de los ejemplos): el modelo
                      debe leer la matriz y calcular la coordenada el mismo.
    """
    t = cargar("navegante", version)
    if not fewshot:
        t = _RE_EJEMPLOS.sub("", t)
    if not observar:
        # 1) lista de vecinos en los ejemplos (si quedan)
        t = _RE_BLOQUE_VECINOS_EJ.sub("", t)
        # 2) lista de vecinos de la situacion actual
        assert "Movimientos posibles:\n{vecinos}\n" in t
        t = t.replace("Movimientos posibles:\n{vecinos}\n", "")
        # 3) instrucciones de la tarea que se apoyaban en esa lista
        assert _FRASE_COPIA in t and _FRASE_VISITADA in t
        t = t.replace(_FRASE_COPIA, _FRASE_COPIA_SIN).replace(_FRASE_VISITADA, _FRASE_VISITADA_SIN)
    return t


def bloque_correccion(motivo, hechos_libres, version="v1", observar=True):
    """Fragmento que se inserta en {bloque_correccion} solo en un reintento."""
    t = cargar("navegante_correccion", version)
    if not observar:
        # sin observar() el Navegante no recibe la lista de celdas libres ni la referencia a ella
        t = re.sub(r"Hechos verificados del entorno: \{hechos_libres\}\n", "", t)
        t = t.replace('Elige otro movimiento usando solo una celda libre de "Movimientos posibles".',
                      "Elige otro movimiento hacia una celda libre.")
    return "\n" + t.format(motivo=motivo, hechos_libres=hechos_libres)


def render_navegante(plantilla, obs, entorno, correccion="", version="v1", observar=True):
    return plantilla.format(
        mapa=obs.mapa, pos_f=entorno.pos[0], pos_c=entorno.pos[1],
        meta_f=entorno.meta[0], meta_c=entorno.meta[1],
        direccion_meta=obs.direccion_meta, historial=obs.historial,
        vecinos=obs.vecinos, bloque_correccion=correccion)


# ---------------------------------------------------------------------------- Critico / Arbitro
def render_critico(hechos_texto, version="v1"):
    return cargar("critico", version).format(hechos=hechos_texto)


def render_arbitro(entorno, candidatos_txt, version="v1"):
    return cargar("arbitro", version).format(
        pos_f=entorno.pos[0], pos_c=entorno.pos[1],
        meta_f=entorno.meta[0], meta_c=entorno.meta[1], candidatos=candidatos_txt)
