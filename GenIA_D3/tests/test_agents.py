import re
import pytest
from src.agents import parse_navegante, parse_critico, parse_arbitro
from src.prompts import plantilla_navegante, bloque_correccion, cargar
import string


def test_navegante_formato_normal():
    p = parse_navegante("Meta hacia: Arriba e Izquierda | Elijo: Izquierda | (5, 5) | valor: 0 | adyacente: sí")
    assert (p.nombre, p.coord, p.valor) == ("Izquierda", (5, 5), 0)


def test_navegante_no_confunde_meta_hacia_con_la_decision():
    p = parse_navegante("Meta hacia: Abajo | Elijo: Derecha | (2, 4) | valor: 0 | adyacente: sí")
    assert p.nombre == "Derecha"
    p = parse_navegante("Meta hacia: Abajo | (2, 4) | valor: 0")          # sin "Elijo": no hay decision con nombre
    assert p.nombre is None and p.coord == (2, 4)


def test_navegante_ruido():
    p = parse_navegante("Claro!\nMeta hacia: Derecha | Elijo: derecha | ( 4 , 5 ) | valor: 0 | adyacente: sí\nPorque...")
    assert (p.nombre, p.coord) == ("Derecha", (4, 5))
    assert parse_navegante("").nombre is None
    assert parse_navegante("Elijo: Arriba | (-1, 5) | valor: 0").coord == (-1, 5)


def test_critico():
    assert parse_critico("Veredicto: APRUEBO").veredicto == "APRUEBO"
    v = parse_critico("Veredicto: CORRIJO | Movimiento: Izquierda | (2, 7)")
    assert (v.veredicto, v.nombre, v.coord) == ("CORRIJO", "Izquierda", (2, 7))
    v = parse_critico("Veredicto: RECHAZO | Motivo: Abajo (4, 3) es un muro (valor 1). Celdas libres: Arriba (2, 3)")
    assert v.veredicto == "RECHAZO" and v.motivo.startswith("Abajo (4, 3) es un muro")
    assert parse_critico("no sé").veredicto is None
    assert parse_critico("veredicto: aprueba").veredicto is None


def test_arbitro():
    assert parse_arbitro("ELIJO: 2 | Abajo | (7, 2) | motivo: x").numero == 2
    assert parse_arbitro("Elijo 1").numero == 1
    assert parse_arbitro("el segundo").numero is None


EXPECTED = {
    "navegante_v1.txt": {"mapa", "pos_f", "pos_c", "meta_f", "meta_c", "direccion_meta", "historial", "vecinos", "bloque_correccion"},
    "navegante_correccion_v1.txt": {"motivo", "hechos_libres"},
    "critico_v1.txt": {"hechos"},
    "arbitro_v1.txt": {"pos_f", "pos_c", "meta_f", "meta_c", "candidatos"},
}


@pytest.mark.parametrize("archivo,esperado", EXPECTED.items())
def test_placeholders(archivo, esperado):
    from src.prompts import DIR_PROMPTS
    t = (DIR_PROMPTS / archivo).read_text(encoding="utf-8")
    assert {f for _, f, _, _ in string.Formatter().parse(t) if f} == esperado


@pytest.mark.parametrize("fewshot", [True, False])
@pytest.mark.parametrize("observar", [True, False])
def test_ablaciones_de_prompt(fewshot, observar):
    t = plantilla_navegante("v1", fewshot, observar)
    assert ("<ejemplos>" in t) == fewshot
    assert ("{vecinos}" in t) == observar
    assert ("COPIA la coordenada" in t) == observar
    if not observar:
        assert "Movimientos posibles" not in t
    bc = bloque_correccion("m", "h", "v1", observar)
    assert ("Hechos verificados" in bc) == observar
    assert ("Movimientos posibles" in bc) == observar
