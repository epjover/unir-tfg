# -*- coding: utf-8 -*-
"""Infraestructura común de evaluación RESPIR-AI (protocolo v2).

Rolling-origin con 36 orígenes en test; para cada origen t0:
  contexto = [t0-C+1, t0] (dentro del segmento), predicción = [t0+1, t0+48],
  recortada a H ∈ {6,12,24,48}. Todos los modelos evalúan las mismas ventanas.

Rutas relativas a la raíz del repositorio (carpeta que contiene este fichero):
  data/biologico-1_clean.parquet  y  results/eval_protocol.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
DATA = str(ROOT / "data" / "biologico-1_clean.parquet")
PROTO = str(ROOT / "results" / "eval_protocol.json")

SEED = 42
HORIZONS = [6, 12, 24, 48]
H_MAX = 48
TARGET = "our"
METEO = ["temperature_2m", "relative_humidity_2m", "precipitation",
         "surface_pressure", "wind_speed_10m", "shortwave_radiation"]
PROC = ["solidos", "sour", "aur", "rn", "trc", "trh_2ppm", "min_real",
        "ox_real", "nh4e", "min_rn_ini", "min_rn_fin", "no3e", "nur"]


def load_all():
    df = pq.read_table(DATA).to_pandas()
    proto = json.load(open(PROTO))
    # covariables de proceso: relleno ffill/bfill DENTRO de cada segmento
    # (huecos internos escasos; nunca se cruza frontera de segmento)
    for c in PROC:
        df[c] = df.groupby("segment_id")[c].transform(
            lambda s: s.ffill().bfill())
    return df, proto


def get_origins(df, proto):
    """Lista de orígenes: (timestamp, segment_id, iloc en df)."""
    out = []
    for o in proto["rolling_origin"]["origenes"]:
        ts = pd.Timestamp(o["timestamp"])
        i = df.index.get_loc(ts)
        assert df["segment_id"].iloc[i] == o["segment_id"]
        out.append({"t0": ts, "segment_id": o["segment_id"], "iloc": i})
    return out


def true_targets(df, origins):
    """y_true con forma (36, 48)."""
    Y = np.stack([df[TARGET].iloc[o["iloc"] + 1:o["iloc"] + 1 + H_MAX].to_numpy()
                  for o in origins])
    assert not np.isnan(Y).any()
    return Y


def split_masks(df, proto):
    m = {}
    for split in ["train", "val", "test"]:
        p = proto["particion"][split]
        m[split] = ((df.index >= p["inicio"]) & (df.index <= p["fin"])
                    & (df["segment_id"] >= 0))
    return m


def metrics_pooled(y_true, y_pred):
    """Métricas agregadas sobre todos los puntos (ventanas apiladas)."""
    yt, yp = y_true.ravel(), y_pred.ravel()
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mape = float(np.mean(np.abs((yt - yp) / yt)) * 100.0)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def evaluate_preds(preds, y_true, modelo, escenario, t_inf_s, contexto,
                   origins, extra=None):
    """preds: (36,48). Devuelve (filas_resumen, filas_por_ventana)."""
    rows, win_rows = [], []
    for H in HORIZONS:
        m = metrics_pooled(y_true[:, :H], preds[:, :H])
        row = {"modelo": modelo, "escenario": escenario, "H": H, **m,
               "t_inferencia_s": t_inf_s, "contexto": contexto}
        if extra:
            row.update(extra)
        rows.append(row)
        for w, o in enumerate(origins):
            win_rows.append({
                "modelo": modelo, "escenario": escenario, "H": H,
                "origen": str(o["t0"]), "ventana": w,
                "MAE": float(np.mean(np.abs(y_true[w, :H] - preds[w, :H])))})
    return rows, win_rows
