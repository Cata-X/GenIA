"""Pruebas del orquestador con LLMs de guion (sin modelo real): cada una fuerza un camino del flujo."""
import re
import pytest

from src.maze import LAB_D1, evaluar_ruta, bfs, construir_dataset
from src.orchestrator import resolver, CONFIGS, Config, Tracer
from src.llm_simulado import LLMSimulado

LAB, A, B = LAB_D1, (0, 0), (0, 8)


class Guion:
    """LLM de guion: `reglas` es una lista de (regex sobre el prompt, respuesta o funcion(prompt)->respuesta).
    Gana la primera regla que calce. Guarda todos los prompts recibidos."""
    def __init__(self, reglas, default="???"):
        self.reglas, self.default, self.prompts = reglas, default, []

    def _resp(self, prompt):
        self.prompts.append(prompt)
        for rx, r in self.reglas:
            if re.search(rx, prompt, re.S):
                return r(prompt) if callable(r) else r
        return self.default

    def generar(self, prompt, max_new_tokens=60, n=1, temperatura=0.0, seed=None):
        return [self._resp(prompt) for _ in range(n if temperatura else 1)]

    def generar_lote(self, prompts, max_new_tokens=60):
        return [self._resp(p) for p in prompts]


def nav(prompt):  # busca la primera celda libre de "Movimientos posibles" de la situacion (a partir de Abajo/Derecha)
    sit = prompt.split("<situacion>")[-1]
    for nombre, f, c in re.findall(r"- (Arriba|Abajo|Izquierda|Derecha): \((\d+), (\d+)\) -> valor 0 \[libre\]", sit):
        return f"Meta hacia: X | Elijo: {nombre} | ({f}, {c}) | valor: 0 | adyacente: sí"


def test_ruta_siempre_legal_y_determinista():
    for nombre, cfg in CONFIGS.items():
        for x in construir_dataset(4):
            lab, a, b = x["laberinto"], tuple(x["inicio"]), tuple(x["meta"])
            r1 = resolver(LLMSimulado(seed=3), lab, a, b, cfg, id_lab=x["id"])
            r2 = resolver(LLMSimulado(seed=3), lab, a, b, cfg, id_lab=x["id"])
            assert r1["ruta"] == r2["ruta"], f"{nombre}: no determinista"
            # la ruta que entrega el orquestador nunca pisa muro ni salta (mover() lo impide)
            estado, _ = evaluar_ruta(lab, a, b, r1["ruta"])
            assert estado in ("exito", "no_llega_meta"), (nombre, estado)
            assert r1["estado"] in ("exito", "limite_pasos", "propuestas_invalidas")


def test_flujo_completo_resuelve_con_modelo_simulado():
    llm = LLMSimulado(seed=0)
    r = resolver(llm, LAB, A, B, CONFIGS["completo"], id_lab=0)
    assert r["estado"] == "exito"
    c = r["contadores"]
    assert c["llamadas_navegante"] >= r["pasos"] and c["llamadas_critico"] > 0


def test_reintento_usa_el_hecho_y_no_la_salida_mala():
    """1er intento: pisa un muro (Abajo de (0,0) es el muro (1,0)). El Critico RECHAZA con motivo. El 2do
    prompt del Navegante debe llevar el motivo + las celdas libres, y NO la salida mala."""
    mala = "Meta hacia: Abajo | Elijo: Abajo | (1, 0) | valor: 0 | adyacente: sí"

    def navegante(prompt):
        if "<correccion>" in prompt:
            return "Meta hacia: Derecha | Elijo: Derecha | (0, 1) | valor: 0 | adyacente: sí"
        return mala

    llm = Guion([(r"Eres el Navegante", navegante),
                 (r"Eres el Crítico", lambda p: "Veredicto: APRUEBO" if "Propuesta del Navegante: Derecha" in p
                  else "Veredicto: RECHAZO | Motivo: Abajo (1, 0) es un muro (valor 1)"),
                 (r"Eres el Árbitro", "ELIJO: 1")])
    cfg = Config(nombre="t", k=1, temperatura=0.0, max_pasos=1)
    r = resolver(llm, LAB, A, B, cfg, id_lab=0)
    assert r["contadores"]["rondas_vacias"] == 1
    assert r["ruta"] == [(0, 0), (0, 1)]
    navs = [p for p in llm.prompts if "Eres el Navegante" in p[:60]]
    assert len(navs) == 2
    segundo = navs[1].split("<situacion>")[-1]
    assert "Abajo (1, 0) es un muro (valor 1)" in segundo           # el motivo del Critico
    assert "las únicas celdas libres vecinas son Derecha (0, 1)" in segundo
    assert mala not in segundo and "Elijo: Abajo | (1, 0)" not in segundo   # no se reinyecta el error


def test_propuestas_invalidas_tras_max_reint():
    llm = Guion([(r"Eres el Navegante", "Meta hacia: Abajo | Elijo: Abajo | (1, 0) | valor: 0 | adyacente: sí"),
                 (r"Eres el Crítico", "Veredicto: RECHAZO | Motivo: muro")])
    r = resolver(llm, LAB, A, B, Config(nombre="t", k=2), id_lab=0)
    assert r["estado"] == "propuestas_invalidas"
    assert r["contadores"]["llamadas_navegante"] == 4          # 1 intento + 3 reintentos
    assert r["pasos"] == 0


def test_critico_que_aprueba_un_muro_no_mueve_al_agente():
    llm = Guion([(r"Eres el Navegante", "Meta hacia: Abajo | Elijo: Abajo | (1, 0) | valor: 0 | adyacente: sí"),
                 (r"Eres el Crítico", "Veredicto: APRUEBO")])
    r = resolver(llm, LAB, A, B, Config(nombre="t", k=1, temperatura=0.0), id_lab=0)
    assert r["estado"] == "propuestas_invalidas" and r["pasos"] == 0
    assert r["contadores"]["critico_errores_graves"] == 4      # una por ronda
    assert r["contadores"]["critico_aciertos"] == 0


def test_critico_ilegible_se_abstiene_y_manda_el_nombre():
    llm = Guion([(r"Eres el Navegante", nav), (r"Eres el Crítico", "mmm no sé")])
    r = resolver(llm, LAB, A, B, Config(nombre="t", k=1, temperatura=0.0, max_pasos=3), id_lab=0)
    assert r["contadores"]["critico_ilegibles"] == r["contadores"]["critico_revisiones"] > 0
    assert r["pasos"] == 3


def test_arbitro_ilegible_cae_a_mayoria_de_votos():
    # En (0,0) del laberinto D1 solo hay 1 vecino libre; hay que llegar a una celda con 2 candidatos: (1,1)
    # -> usamos un LLM que alterna propuestas para forzar 2 candidatos y un Arbitro que responde basura.
    llm = LLMSimulado(seed=5)
    llm_arb = Guion([(r"Eres el Árbitro", "no sé")])
    class Mixto:
        def generar(self, prompt, *a, **k):
            return (llm_arb if "Eres el Árbitro" in prompt[:60] else llm).generar(prompt, *a, **k)
        def generar_lote(self, prompts, *a, **k):
            return [self.generar(p)[0] for p in prompts]
    r = resolver(Mixto(), LAB, A, B, CONFIGS["completo"], id_lab=0)
    c = r["contadores"]
    assert c["arbitro_decisiones"] == c["arbitro_fallbacks"] == c["llamadas_arbitro"]
    assert c["arbitro_decisiones"] > 0


def test_ablaciones_apagan_lo_que_dicen():
    llm = LLMSimulado(seed=1)
    c = resolver(llm, LAB, A, B, CONFIGS["sin_critico"], id_lab=0)["contadores"]
    assert c["llamadas_critico"] == 0 and c["critico_revisiones"] == 0
    c = resolver(LLMSimulado(seed=1), LAB, A, B, CONFIGS["sin_selfconsistency"], id_lab=0)["contadores"]
    assert c["llamadas_arbitro"] == 0 and c["propuestas"] == c["llamadas_navegante"]       # k = 1
    c = resolver(LLMSimulado(seed=1), LAB, A, B, CONFIGS["sin_arbitro"], id_lab=0)["contadores"]
    assert c["llamadas_arbitro"] == 0
    c = resolver(LLMSimulado(seed=1), LAB, A, B, CONFIGS["arbitro_regla"], id_lab=0)["contadores"]
    assert c["llamadas_arbitro"] == 0


def test_traza_tiene_los_mensajes_entre_agentes():
    tr = Tracer(guardar_prompts=True)
    resolver(LLMSimulado(seed=0), LAB, A, B, CONFIGS["completo"], id_lab=0, tracer=tr)
    pares = {(e["de"], e["a"]) for e in tr.eventos if e["tipo"] == "mensaje"}
    assert {("Navegante", "Orquestador"), ("Orquestador", "Crítico"), ("Crítico", "Orquestador"),
            ("Orquestador", "Entorno")} <= pares
    assert any("prompt" in e for e in tr.eventos if e.get("a") == "Navegante")
