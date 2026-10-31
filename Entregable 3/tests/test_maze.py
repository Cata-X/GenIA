import json
from pathlib import Path
from src.maze import construir_dataset, bfs, evaluar_ruta, LAB_D1

RAIZ = Path(__file__).resolve().parent.parent


def test_dataset_guardado_coincide_con_regenerado():
    guardado = json.loads((RAIZ / "data" / "laberintos_9x9.json").read_text(encoding="utf-8"))
    regenerado = json.loads(json.dumps(construir_dataset(20)))
    assert guardado == regenerado
    assert len(guardado) == 20 and guardado[0]["laberinto"] == LAB_D1


def test_todos_resolubles_y_dist_minima():
    for x in construir_dataset(20):
        camino = bfs(x["laberinto"], tuple(x["inicio"]), tuple(x["meta"]))
        assert camino is not None
        assert x["id"] == 0 or len(camino) - 1 >= 10


def test_verificador():
    lab, a, b = LAB_D1, (0, 0), (0, 8)
    ruta = bfs(lab, a, b)
    assert evaluar_ruta(lab, a, b, ruta)[0] == "exito"
    assert evaluar_ruta(lab, a, b, [(0, 0), (0, 3)])[0] == "salto_no_adyacente"
    assert evaluar_ruta(lab, a, b, [(0, 0), (1, 0)])[0] == "pisa_muro"
    assert evaluar_ruta(lab, a, b, [(0, 0), (-1, 0)])[0] == "fuera_de_rango"
    assert evaluar_ruta(lab, a, b, [(0, 0), (0, 1)])[0] == "no_llega_meta"
