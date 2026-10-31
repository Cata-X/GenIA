"""Orquestador del sistema multiagente (generar - validar - elegir).

Por cada paso del recorrido:

    1. observar()                          herramienta   -> situacion
    2. Navegante x k (T=0.7)               agente        -> k propuestas        [self-consistency]
    3. validar() + Critico                 herramienta + agente -> APRUEBO / CORRIJO / RECHAZO por propuesta
    4. si ningun candidato sobrevive       -> reintento: el Navegante recibe el HECHO que dijo el
       Critico (no su propia salida mala), hasta max_reint veces              [Self-Refine, aislamiento]
    5. Arbitro                             agente        -> elige entre candidatos validos (si hay >1)
    6. mover()                             herramienta   -> unica forma de avanzar; rechaza lo ilegal

Corte: meta alcanzada / max_pasos / max_reint reintentos seguidos sin candidato ("propuestas_invalidas").
Cada flag de `Config` apaga exactamente una tecnica (ver CONFIGS) para las ablaciones.
"""
import json
import time
from dataclasses import dataclass, asdict, replace
from pathlib import Path

from .maze import evaluar_ruta
from .tools import Entorno
from .prompts import plantilla_navegante, bloque_correccion
from .agents import Navegante, Critico, Arbitro, clave_regla


# ============================================================================ configuracion
@dataclass(frozen=True)
class Config:
    nombre: str = "completo"
    fewshot: bool = True          # Prompting: ejemplos positivos en el prompt del Navegante
    observar: bool = True         # Context (select): lista de vecinos con coordenada / marca de visitada
    k: int = 3                    # Reasoning: self-consistency, k propuestas por ronda (k=1 -> greedy)
    temperatura: float = 0.7
    critico: bool = True          # Structure/Refine: Critico-Corrector con hechos del entorno
    arbitro: str = "llm"          # "llm" | "votos" (mayoria) | "regla" (los mismos criterios, en codigo)
    max_reint: int = 3            # reintentos del Navegante en un paso sin candidatos
    max_pasos: int = 81           # igual que el D2 (9x9 celdas)
    max_tokens_nav: int = 60
    max_tokens_cri: int = 80
    max_tokens_arb: int = 40
    version_prompts: str = "v1"
    seed: int = 2026

    def a_dict(self):
        return asdict(self)


# Tecnica que se apaga en cada ablacion (una a la vez, todo lo demas igual a `completo`).
CONFIGS = {
    "completo":             Config(nombre="completo"),
    "sin_fewshot":          Config(nombre="sin_fewshot", fewshot=False),
    "sin_observar":         Config(nombre="sin_observar", observar=False),
    "sin_selfconsistency":  Config(nombre="sin_selfconsistency", k=1, temperatura=0.0),
    "sin_critico":          Config(nombre="sin_critico", critico=False),
    "sin_arbitro":          Config(nombre="sin_arbitro", arbitro="votos"),
    # extra, para saber si el LLM aporta algo frente a la misma regla escrita en codigo
    "arbitro_regla":        Config(nombre="arbitro_regla", arbitro="regla"),
}


# ============================================================================ trazas
class Tracer:
    """Registra los mensajes entre agentes y herramientas. `eventos` se vuelca a JSONL con `volcar()`
    al terminar el laberinto (todo o nada: una corrida interrumpida no deja trazas a medias)."""

    def __init__(self, path=None, verbose=False, guardar_prompts=False, ancho=150):
        self.path, self.verbose, self.guardar_prompts, self.ancho = path, verbose, guardar_prompts, ancho
        self.eventos = []
        self.contexto = {}

    def contexto_de(self, **kw):
        self.contexto = kw

    def msg(self, de, a, contenido, prompt=None, **extra):
        ev = {"tipo": "mensaje", **self.contexto, "de": de, "a": a, "contenido": contenido, **extra}
        if prompt is not None and self.guardar_prompts:
            ev["prompt"] = prompt
        self.eventos.append(ev)
        if self.verbose:
            txt = " / ".join(l.strip() for l in str(contenido).splitlines() if l.strip())
            txt = txt if len(txt) <= self.ancho else txt[: self.ancho - 1] + "…"
            print(f"      {de} → {a}: {txt}")

    def nota(self, texto, **extra):
        self.eventos.append({"tipo": "nota", **self.contexto, "texto": texto, **extra})
        if self.verbose:
            print(texto)

    def volcar(self):
        if self.path and self.eventos:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                for ev in self.eventos:
                    f.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")


def semilla(cfg, id_lab, paso, ronda):
    return cfg.seed * 1_000_003 + int(id_lab) * 10_007 + paso * 101 + ronda


# ============================================================================ nucleo
def resolver(llm, lab, inicio, meta, cfg=None, id_lab=0, tracer=None):
    """Resuelve un laberinto con el sistema multiagente. Devuelve un dict con las mismas claves que el
    D2 (estado, ruta, pasos, invalidas, valor_mal_declarado, coordenada_mal_declarada, tiempo_s)
    mas contadores propios del D3 (ver `m`)."""
    cfg = cfg or CONFIGS["completo"]
    tr = tracer or Tracer()
    t0 = time.time()
    env = Entorno(lab, inicio, meta)

    nav = Navegante(llm, plantilla_navegante(cfg.version_prompts, cfg.fewshot, cfg.observar),
                    cfg.version_prompts, cfg.observar, cfg.max_tokens_nav)
    cri = Critico(llm, cfg.version_prompts, cfg.max_tokens_cri) if cfg.critico else None
    arb = Arbitro(llm, cfg.version_prompts, cfg.max_tokens_arb) if cfg.arbitro == "llm" else None

    m = dict(llamadas_navegante=0, llamadas_critico=0, llamadas_arbitro=0, propuestas=0,
             invalidas=0, valor_mal_declarado=0, coordenada_mal_declarada=0, rondas_vacias=0,
             critico_revisiones=0, critico_aciertos=0, critico_correcciones=0, critico_rechazos=0,
             critico_ilegibles=0, critico_errores_graves=0,
             arbitro_decisiones=0, arbitro_cambia_voto=0, arbitro_fallbacks=0)
    estado = None
    paso = 0

    while env.pos != env.meta and len(env.ruta) - 1 < cfg.max_pasos:
        paso += 1
        obs = env.observar()
        tr.contexto_de(id=id_lab, config=cfg.nombre, paso=paso, pos=env.pos)
        if tr.verbose:
            print(f"\n  [paso {paso}] posición {env.pos} → meta {env.meta} ({obs.direccion_meta})")
        correccion = ""
        cands = []
        for ronda in range(cfg.max_reint + 1):
            tr.contexto_de(id=id_lab, config=cfg.nombre, paso=paso, pos=env.pos, ronda=ronda)
            prompt = nav.prompt(obs, env, correccion)
            tr.msg("Orquestador", "Navegante",
                   ("(reintento con corrección) " + correccion.strip().replace("\n", " ")) if correccion
                   else f"situación en {env.pos}", prompt=prompt)
            props = nav.proponer(prompt, cfg.k, cfg.temperatura, semilla(cfg, id_lab, paso, ronda))
            m["llamadas_navegante"] += 1
            for p in props:
                tr.msg("Navegante", "Orquestador", p.crudo)
            cands, motivos = _evaluar_propuestas(env, props, cri, cfg, m, tr)
            if cands:
                break
            m["rondas_vacias"] += 1
            motivo = next((x for x in motivos if x), None) or "ninguna propuesta llevaba a una celda libre"
            correccion = bloque_correccion(motivo, f"las únicas celdas libres vecinas son {env.texto_libres(env.libres())}",
                                           cfg.version_prompts, cfg.observar)
            tr.nota(f"      ✗ ronda {ronda}: sin candidatos válidos — {motivo}")
        if not cands:
            estado = "propuestas_invalidas"
            break

        elegido = _elegir(env, cands, arb, cfg, m, tr)
        env.mover(elegido["celda"])
        tr.msg("Orquestador", "Entorno", f"mover({elegido['nombre']}) → {elegido['celda']}")

    ruta_final = env.ruta
    if estado is None:
        estado, ruta_final = evaluar_ruta(lab, inicio, meta, env.ruta)
        if estado == "no_llega_meta":
            estado = "limite_pasos"
    exito = estado == "exito"
    tr.contexto_de(id=id_lab, config=cfg.nombre)
    tr.nota(f"  → resultado: {estado} en {len(ruta_final) - 1} pasos", tipo_evento="fin", estado=estado)
    return {"estado": estado, "ruta": ruta_final, "pasos": len(ruta_final) - 1,
            "invalidas": m["invalidas"], "valor_mal_declarado": m["valor_mal_declarado"],
            "coordenada_mal_declarada": m["coordenada_mal_declarada"],
            "exito_estricto": exito and m["rondas_vacias"] == 0,
            "tiempo_s": round(time.time() - t0, 1), "contadores": m,
            "traza": tr.eventos}


# ============================================================================ pasos internos
def _evaluar_propuestas(env, props, cri, cfg, m, tr):
    """Convierte las propuestas del Navegante en candidatos legales (con votos) y reune motivos de rechazo."""
    info = []                      # una entrada por propuesta
    for p in props:
        m["propuestas"] += 1
        nc = env.celda_de(p.nombre) if p.nombre else None
        legal = nc is not None and env.valor(nc) == 0
        if not legal:
            m["invalidas"] += 1
        if nc is not None and p.valor is not None and p.valor != env.valor(nc):
            m["valor_mal_declarado"] += 1
        if p.nombre and p.coord is not None and p.coord != nc:
            m["coordenada_mal_declarada"] += 1
        info.append({"p": p, "hechos": env.validar(p.nombre, p.coord, p.valor) if p.nombre else None,
                     "celda": None, "motivo": None})

    veredictos = {}
    if cri is not None:
        unicos = {}
        for it in info:
            if it["hechos"] is not None:
                unicos.setdefault(it["hechos"].texto, it["hechos"])
        textos = list(unicos)
        if textos:
            prompts = [cri.prompt(t) for t in textos]
            for t, pr in zip(textos, prompts):
                tr.msg("Orquestador", "Crítico", t, prompt=pr)
            vs = cri.revisar_lote(prompts)
            m["llamadas_critico"] += len(prompts)
            for t, v in zip(textos, vs):
                veredictos[t] = v
                tr.msg("Crítico", "Orquestador", v.crudo)

    for it in info:
        p, h = it["p"], it["hechos"]
        if h is None:                                   # el Navegante no escribio un movimiento reconocible
            it["motivo"] = env.motivo_sin_nombre()
            continue
        v = veredictos.get(h.texto)
        if cri is None or v is None:
            _resolver_mecanico(it, h)
            continue
        m["critico_revisiones"] += 1
        esperado = h.esperado
        if v.veredicto is None:                         # el Critico no respondio en el formato: se abstiene
            m["critico_ilegibles"] += 1
            _resolver_mecanico(it, h)
            continue
        if v.veredicto == "APRUEBO":
            acierto = esperado[0] == "APRUEBO"
            if h.nombre_libre:
                it["celda"] = h.celda_nombre
            else:
                m["critico_errores_graves"] += 1        # aprobo un movimiento ilegal: lo frena mover()/la herramienta
                it["motivo"] = h.motivo
        elif v.veredicto == "CORRIJO":
            m["critico_correcciones"] += 1
            nombre = v.nombre or (env.nombre_de(v.coord) if v.coord else None)
            c = env.celda_de(nombre) if nombre else None
            acierto = esperado[0] == "CORRIJO" and c is not None and c == esperado[2]
            if c is not None and env.valor(c) == 0:
                it["celda"] = c
            else:
                m["critico_errores_graves"] += 1
                it["motivo"] = h.motivo or "la corrección del Crítico no llevaba a una celda libre"
        else:                                           # RECHAZO
            m["critico_rechazos"] += 1
            acierto = esperado[0] == "RECHAZO"
            it["motivo"] = v.motivo or h.motivo or "el Crítico rechazó la propuesta"
        m["critico_aciertos"] += int(acierto)

    votos, orden = {}, []
    for it in info:
        c = it["celda"]
        if c is not None:
            if c not in votos:
                votos[c] = 0
                orden.append(c)
            votos[c] += 1
    cands = [env.candidato(c, votos[c]) for c in orden]
    return cands, [it["motivo"] for it in info]


def _resolver_mecanico(it, h):
    """Sin Critico (o si se abstiene): manda el NOMBRE, como en el agente del D2; la herramienta solo
    comprueba que la celda sea legal."""
    if h.nombre_libre:
        it["celda"] = h.celda_nombre
    else:
        it["motivo"] = h.motivo


def _voto(cands):
    """Mayoria de votos; en empate, el primero que aparecio."""
    return max(range(len(cands)), key=lambda i: (cands[i]["votos"], -i))


def _elegir(env, cands, arb, cfg, m, tr):
    if len(cands) == 1:
        return cands[0]
    m["arbitro_decisiones"] += 1
    por_voto = _voto(cands)
    texto = env.texto_candidatos(cands)
    if cfg.arbitro == "llm":
        pr = arb.prompt(env, texto)
        tr.msg("Orquestador", "Árbitro", texto, prompt=pr)
        e = arb.elegir(pr)
        m["llamadas_arbitro"] += 1
        tr.msg("Árbitro", "Orquestador", e.crudo)
        if e.numero is not None and 1 <= e.numero <= len(cands):
            idx = e.numero - 1
        else:
            m["arbitro_fallbacks"] += 1
            idx = por_voto
            tr.nota("      (Árbitro ilegible: se usa la mayoría de votos)")
    elif cfg.arbitro == "regla":
        idx = min(range(len(cands)), key=lambda i: (clave_regla(cands[i], env.meta), i))
        tr.msg("Orquestador", "Regla", f"{texto}  →  elige {idx + 1}")
    else:                                              # "votos"
        idx = por_voto
        tr.msg("Orquestador", "Votos", f"{texto}  →  elige {idx + 1}")
    if idx != por_voto:
        m["arbitro_cambia_voto"] += 1
    return cands[idx]
