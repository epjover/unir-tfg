# RESPIR-AI — Predicción de la OUR horaria en una EDAR (TFG)

Comparativa de un modelo clásico (XGBoost + skforecast) frente a dos modelos fundacionales de
series temporales (Amazon **Chronos-2** y IBM **Granite TTM r2**) para predecir la tasa de consumo
de oxígeno (OUR) del reactor biológico 1, con estudios de contexto, covariables y regímenes de
adaptación.

## Estructura

```
repo/
├── common_eval.py        # módulo compartido: carga, orígenes rolling-origin, métricas (protocolo v2)
├── requirements.txt
├── data/                 # biologico-1.parquet (crudo, NO distribuido), biologico-1_clean.parquet (NO distribuido)
├── results/              # eval_protocol.json + CSV/parquet de resultados (SÍ incluidos)
├── figures/              # figuras PNG (SÍ incluidas)
└── notebooks/            # 01..14, en español, con salidas ejecutadas
```

## Orden de ejecución

| Nº | Cuaderno | GPU | Tiempo aprox. |
|---|---|---|---|
| 01 | EDA y limpieza | no | 1 min |
| 02 | Particionado y protocolo v2 | no | segundos |
| 03 | Baseline XGBoost (E1–E3) | no | 5–10 min |
| 04 | Chronos-2 (E1–E6) | sí | ~1 h |
| 05 | TTM (E1–E4) | sí | ~15 min |
| 06 | Comparativa y tests estadísticos | no | 1 min |
| 07 | Barrido contexto × covariables | sí | ~2 h |
| 08 | Contexto extendido (hasta 8192 h) | sí | ~1 h |
| 09 | Coste de adaptación (full FT / 100 %) | sí | ~10 min |
| 10 | Covariables de calendario | sí | ~5 min |
| 11 | Mejoras de bajo coste (directo, sesgo, ensembles) | sí | ~5 min |
| 12 | Validación por bloques mensuales (forward-chaining) | sí | ~10 min |
| 13 | Rigor inferencial (selección en validación, Holm/BH, MASE, semillas, bootstrap, auditoría de fugas) | sí | ~10 min |
| 14 | Cobertura de intervalos (cuantiles nativos vs conformal, interval score, CRPS) | sí | ~10 min |

Los cuadernos 04, 05, 07 y 08 incluyen al final una celda **«Resultados guardados»** que carga las
tablas ya calculadas desde `results/` — ejecutable sin GPU. Los cuadernos 01, 02, 03, 06, 09 y 10 se
distribuyen con todas sus salidas ejecutadas.

## Requisitos

- Python 3.11, `pip install -r requirements.txt`.
- GPU con ≥8 GB de VRAM para 04/05/07/08/09/10 (todo el estudio se ejecutó en una RTX 3070 de 8 GB).
- Los pesos de `amazon/chronos-2` (~460 MB) y `ibm-granite/granite-timeseries-ttm-r2` (~10 MB) se
  descargan de Hugging Face en el primer uso.

## Datos y confidencialidad

`data/biologico-1.parquet` (datos crudos de planta) y su derivado `data/biologico-1_clean.parquet`
**no se distribuyen** por confidencialidad (datos de proceso de Sensara). Las variables
meteorológicas proceden de Open-Meteo (archivo histórico, licencia abierta).

**Qué se puede reproducir sin los datos**: el cuaderno 06 (comparativa, Diebold-Mariano, Friedman)
y todas las figuras de resultados, porque parten exclusivamente de los CSV/parquet incluidos en
`results/`. Con los datos en `data/`, la cadena completa 01→14 es reproducible de principio a fin
(semilla 42; los entrenamientos en GPU pueden variar en la 3ª cifra decimal del MAE por
no-determinismo de CUDA).

## Resultados principales

- **H=6 h**: gana el baseline XGBoost univariante (MAE 3.43).
- **H=24 h**: gana TTM few-shot 5 % + meteo futuras (MAE 3.78).
- **H=48 h**: gana Chronos-2 zero-shot C=256 + meteo futuras (MAE 4.02).
- El fine-tuning completo de Chronos-2 **degrada** el zero-shot (significativo en univariante);
  la adaptación ligera (LoRA / few-shot 5 %) es el punto dulce coste/beneficio.
- Las covariables pasadas de proceso y el calendario no aportan mejoras significativas;
  la meteo futura (pronóstico) sí, especialmente en horizonte largo.
- El **ensemble de contextos** de Chronos-2 (media de C ∈ {128,256,512}) es la única mejora de bajo
  coste que supera al mejor modelo individual (MAE@48 = 3.98); la corrección de sesgo de 24 h
  degrada de forma significativa (cuaderno 11).
- La validación por bloques sobre toda la serie (188 ventanas, 14 meses) da **significación** a la
  ventaja de Chronos-2 ZS F sobre XGBoost a H≥12 (DM-HLN p ≤ 0.03) y muestra que el few-shot de TTM
  empeora al zero-shot cuando la historia de entrenamiento es corta (cuaderno 12).
- Rigor (cuaderno 13): con selección honesta en validación la cifra recomendada es MAE@48 = 4.02
  (ensemble de contextos) / 4.15 (individual); tras corrección Holm/BH global sobreviven los efectos
  de bloques y el daño del full FT, mientras la ventaja sobre XGBoost a H=48 queda en el borde
  (DM p≈0.08–0.10, IC bootstrap excluye el 0); el few-shot de TTM no es robusto entre semillas;
  MASE 0.70–0.88 frente al naive estacional; sin fugas de imputación (auditoría de 224 ventanas).
- Intervalos (cuaderno 14): todos los modelos sub-cubren con cuantiles nativos (80 % nominal →
  68–76 % real); el conformal split lo corrige al nivel 80 % (75–80 %); los cuantiles nativos de
  Chronos-2 son los mejor calibrados y el déficit se concentra en el cambio de régimen de noviembre.
