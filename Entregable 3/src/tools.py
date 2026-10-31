"""Herramientas del entorno (las usa el orquestador; los agentes nunca tocan el laberinto directamente).

    observar()  -> lo que ve el Navegante: mapa, vecinos con coordenada y valor, direccion de la meta,
                   ultimos pasos y marcas "(ya visitada)".             [Context engineering: select]
    validar()   -> hechos verificados para el Critico (texto de 6 lineas) + veredicto esperado
                   (verdad de referencia, solo para medir al Critico, nunca se le entrega).
    memoria     -> `visitadas` y `ruta` (el historial que ve el Navegante).   [Context engineering: write]
    mover()     -> unica forma de cambiar la posicion; rechaza cualquier movimiento ilegal.

Los formatos de texto replican EXACTAMENTE los ejemplos de prompts/*.txt (ver prompts/README.md);
tests/test_formatos.py lo comprueba.
"""
from dataclasses import dataclass, field

from .maze import DIRS
from .d2_legacy import direccion_meta  # misma funcion que uso el D2

NOMBRES = [d[0] for d in DIRS]                      # Arriba, Abajo, Izquierda, Derecha
DELTA = {nombre: (dr, dc) for nombre, dr, dc in DIRS}


class MovimientoIlegal(Exception):
    pass


@dataclass
class Observacion:
    mapa: str
    vecinos: str
    direccion_meta: str
    historial: str
    libres: list = field(default_factory=list)       # [(nombre, (f, c)), ...] en orden Arriba..Derecha


@dataclass
class Hechos:
    texto: str                    # las seis lineas que lee el Critico
    libres_txt: str               # "Arriba (f, c), Izquierda (f, c)"  (tambien se usa en el reintento)
    celda_nombre: tuple | None    # celda a la que apunta el NOMBRE (None si cae fuera del mapa)
    nombre_libre: bool            # esa celda existe y es libre
    coord_valida: bool            # la COORDENADA es un vecino dentro del mapa y libre
    coincide: bool
    esperado: tuple               # (veredicto, nombre, celda) segun las reglas del prompt del Critico
    motivo: str                   # frase de error generada por la herramienta (para reintentos sin Critico)


class Entorno:
    def __init__(self, lab, inicio, meta):
        self.lab = lab
        self.n = len(lab)
        self.inicio = tuple(inicio)
        self.meta = tuple(meta)
        self.pos = self.inicio
        self.ruta = [self.inicio]
        self.visitadas = {self.inicio}

    # ------------------------------------------------------------------ utilidades
    def en_mapa(self, celda):
        return celda is not None and 0 <= celda[0] < self.n and 0 <= celda[1] < self.n

    def celda_de(self, nombre, pos=None):
        """Celda a la que lleva `nombre` desde `pos`; None si cae fuera del mapa."""
        pos = pos or self.pos
        dr, dc = DELTA[nombre]
        c = (pos[0] + dr, pos[1] + dc)
        return c if self.en_mapa(c) else None

    def valor(self, celda):
        return self.lab[celda[0]][celda[1]]

    def nombre_de(self, celda, pos=None):
        """Nombre del movimiento que lleva de `pos` a `celda`, o None si no es vecino."""
        pos = pos or self.pos
        for nombre in NOMBRES:
            if self.celda_de(nombre, pos) == tuple(celda):
                return nombre
        return None

    def libres(self, pos=None):
        pos = pos or self.pos
        out = []
        for nombre in NOMBRES:
            c = self.celda_de(nombre, pos)
            if c is not None and self.valor(c) == 0:
                out.append((nombre, c))
        return out

    @staticmethod
    def texto_libres(libres):
        return ", ".join(f"{n} ({c[0]}, {c[1]})" for n, c in libres) or "ninguna"

    # ------------------------------------------------------------------ observar()
    def observar(self):
        lineas = []
        for nombre in NOMBRES:
            c = self.celda_de(nombre)
            if c is None:
                lineas.append(f"- {nombre}: fuera del mapa")
                continue
            v = self.valor(c)
            linea = f"- {nombre}: ({c[0]}, {c[1]}) -> valor {v} [{'muro' if v == 1 else 'libre'}]"
            if v == 0 and c in self.visitadas:
                linea += " (ya visitada)"
            lineas.append(linea)
        historial = " -> ".join(f"({r}, {c})" for r, c in self.ruta[-8:])
        return Observacion(
            mapa="\n".join(str(f) for f in self.lab),
            vecinos="\n".join(lineas),
            direccion_meta=direccion_meta(self.pos, self.meta),
            historial=historial,
            libres=self.libres())

    # ------------------------------------------------------------------ validar()
    def validar(self, nombre, coord, valor_decl):
        """Hechos verificados sobre una propuesta (nombre, coord, valor declarado). `nombre` es obligatorio."""
        pos = self.pos
        libres = self.libres()
        libres_txt = self.texto_libres(libres)
        nc = self.celda_de(nombre)
        coord = tuple(coord) if coord is not None else None

        # linea 2
        coord_txt = f"({coord[0]}, {coord[1]})" if coord is not None else "(sin coordenada)"
        l2 = f"Propuesta del Navegante: {nombre} | {coord_txt} | valor declarado: {valor_decl if valor_decl is not None else 'no indicado'}"
        # linea 3
        if nc is None:
            l3 = f'Celda que corresponde al nombre "{nombre}": fuera del mapa'
            nombre_libre = False
            motivo = f"{nombre} está fuera del mapa"
        else:
            v = self.valor(nc)
            l3 = f'Celda que corresponde al nombre "{nombre}": ({nc[0]}, {nc[1]}), valor {v}, {"libre" if v == 0 else "muro"}'
            nombre_libre = v == 0
            motivo = "" if nombre_libre else f"{nombre} ({nc[0]}, {nc[1]}) es un muro (valor 1)"
        # linea 4
        coincide = coord is not None and nc is not None and coord == nc
        if coincide:
            l4 = "Nombre y coordenada coinciden: sí"
        elif coord is None:
            l4 = "Nombre y coordenada coinciden: no (el Navegante no indicó una coordenada)"
        elif coord == pos:
            l4 = f"Nombre y coordenada coinciden: no (la coordenada ({coord[0]}, {coord[1]}) es la posición actual, no un vecino)"
        elif self.nombre_de(coord) is not None:
            l4 = f'Nombre y coordenada coinciden: no (la coordenada ({coord[0]}, {coord[1]}) corresponde a "{self.nombre_de(coord)}")'
        else:
            l4 = f"Nombre y coordenada coinciden: no (la coordenada ({coord[0]}, {coord[1]}) no es un vecino)"
        # linea 5
        coord_valida = False
        if coord is None:
            l5 = "Coordenada propuesta: no indicada"
        elif not self.en_mapa(coord):
            l5 = "Coordenada propuesta: fuera del mapa"
        elif abs(coord[0] - pos[0]) + abs(coord[1] - pos[1]) != 1:
            l5 = "Coordenada propuesta: no es adyacente"
        else:
            vc = self.valor(coord)
            l5 = f"Coordenada propuesta: valor real {vc}, adyacente: sí"
            coord_valida = vc == 0
        l6 = f"Celdas libres vecinas: {libres_txt}"
        texto = "\n".join([f"Posición: ({pos[0]}, {pos[1]})", l2, l3, l4, l5, l6])

        # veredicto esperado (las mismas reglas que lee el Critico en prompts/critico_v1.txt)
        if coincide and nombre_libre:
            esperado = ("APRUEBO", nombre, nc)
        elif coord_valida:                                  # la coordenada manda
            esperado = ("CORRIJO", self.nombre_de(coord), coord)
        elif nombre_libre:                                  # si no, manda el nombre
            esperado = ("CORRIJO", nombre, nc)
        else:
            esperado = ("RECHAZO", None, None)
        return Hechos(texto, libres_txt, nc, nombre_libre, coord_valida, coincide, esperado, motivo)

    def motivo_sin_nombre(self):
        return "tu respuesta no indicó un movimiento válido (Arriba, Abajo, Izquierda o Derecha)"

    # ------------------------------------------------------------------ candidatos (para el Arbitro)
    def candidato(self, celda, votos):
        """Estadisticas de un candidato legal. `salidas` no cuenta la celda actual."""
        salidas = sum(1 for _, c in self.libres(celda) if c != self.pos)
        return {
            "nombre": self.nombre_de(celda),
            "celda": tuple(celda),
            "visitada": tuple(celda) in self.visitadas,
            "distancia": abs(celda[0] - self.meta[0]) + abs(celda[1] - self.meta[1]),
            "salidas": salidas,
            "votos": votos,
        }

    @staticmethod
    def texto_candidatos(candidatos):
        return "\n".join(
            f"{i}. {c['nombre']} ({c['celda'][0]}, {c['celda'][1]}): "
            f"{'visitada' if c['visitada'] else 'no visitada'} | distancia a la meta: {c['distancia']} | "
            f"salidas: {c['salidas']} | votos: {c['votos']}"
            for i, c in enumerate(candidatos, 1))

    def es_meta(self, celda):
        return tuple(celda) == self.meta

    # ------------------------------------------------------------------ mover()
    def mover(self, celda):
        celda = tuple(celda)
        if self.nombre_de(celda) is None:
            raise MovimientoIlegal(f"{celda} no es adyacente a {self.pos}")
        if self.valor(celda) == 1:
            raise MovimientoIlegal(f"{celda} es un muro")
        self.pos = celda
        self.ruta.append(celda)
        self.visitadas.add(celda)
        return celda
