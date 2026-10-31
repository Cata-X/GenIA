import csv
import pandas as pd
from src.evaluate import evaluar, importar_d2, hechos_previos, COLUMNAS_D2, ORDEN
from src.analisis import resumen, taxonomia, wilson, mcnemar_exacto, cargar, critico, caso_fallo
from src.llm_simulado import LLMSimulado


def test_evaluar_es_reanudable(tmp_path):
    out, tr = tmp_path / "res", tmp_path / "tr"
    ds = tmp_path / "lab.json"
    metodos = ["aleatorio", "completo", "sin_critico"]
    p = evaluar(LLMSimulado(seed=0), metodos, 3, out, tr, ds)
    assert len(hechos_previos(p)) == 9
    evaluar(LLMSimulado(seed=0), metodos, 3, out, tr, ds)        # no repite nada
    assert len(list(csv.DictReader(open(p, encoding="utf-8")))) == 9
    n_lineas = sum(1 for _ in open(tr / "completo.jsonl", encoding="utf-8"))
    evaluar(LLMSimulado(seed=0), metodos, 3, out, tr, ds)
    assert sum(1 for _ in open(tr / "completo.jsonl", encoding="utf-8")) == n_lineas   # trazas sin duplicar
    evaluar(LLMSimulado(seed=0), metodos, 4, out, tr, ds)        # agrega solo el laberinto nuevo
    assert len(hechos_previos(p)) == 12


def test_analisis(tmp_path):
    p = evaluar(LLMSimulado(seed=0), ORDEN, 4, tmp_path / "r", tmp_path / "t", tmp_path / "l.json")
    df = cargar(p)
    r = resumen(df)
    assert list(r.metodo) == ORDEN and (r.n == 4).all()
    assert "p_mcnemar" in r and r.loc[r.metodo == "completo", "tasa"].iloc[0] >= 0
    assert taxonomia(df).sum().sum() == 4 * len(ORDEN)
    assert not critico(df).empty
    caso_fallo(df, tmp_path / "t", metodo="aleatorio")           # el aleatorio siempre falla en el sim
    assert (tmp_path / "t" / "caso_fallo.jsonl").exists()


def test_importar_d2(tmp_path):
    d2 = pd.DataFrame([
        {"id": 0, "metodo": "baseline", "estado": "salto_no_adyacente", "exito": False, "exito_estricto": False, "pasos": 3, "optimo": 12,
         "invalidas": None, "valor_mal_declarado": None, "coordenada_mal_declarada": None, "tiempo_s": 5.0},
        {"id": 0, "metodo": "agente", "estado": "exito", "exito": True, "exito_estricto": False, "pasos": 20, "optimo": 12,
         "invalidas": 4, "valor_mal_declarado": 1, "coordenada_mal_declarada": 3, "tiempo_s": 30.0},
        {"id": 0, "metodo": "agente_sin_memoria", "estado": "limite_pasos", "exito": False, "exito_estricto": False, "pasos": 81, "optimo": 12,
         "invalidas": 9, "valor_mal_declarado": 0, "coordenada_mal_declarada": 0, "tiempo_s": 60.0}])
    d2p = tmp_path / "d2.csv"; d2.to_csv(d2p, index=False)
    p = importar_d2(d2p, tmp_path / "r")
    assert hechos_previos(p) == {(0, "baseline"), (0, "d2_agente")}      # agente_sin_memoria no se importa
    importar_d2(d2p, tmp_path / "r")
    assert len(hechos_previos(p)) == 2


def test_estadistica():
    lo, hi = wilson(2, 20)
    assert 0.02 < lo < 0.10 < hi < 0.32
    import pandas as pd
    a = pd.Series([True] * 10 + [False] * 10, index=range(20))
    b = pd.Series([False] * 10 + [True] * 10, index=range(20))
    assert mcnemar_exacto(a, a) == (0, 0, 1.0)
    bb, cc, pv = mcnemar_exacto(a, b)
    assert (bb, cc) == (10, 10) and pv == 1.0
    assert mcnemar_exacto(a, pd.Series([False] * 20, index=range(20)))[2] < 0.01
