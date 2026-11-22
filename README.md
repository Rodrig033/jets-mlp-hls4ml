# MLP para identificación de jets del LHC (hls4ml) — Segundo parcial, Redes Neuronales I

Autor: Rodrigo Farid Córdoba · Universidad de Xalapa

Clasificación supervisada de 5 clases de jets (g, q, W, Z, t) a partir de 16
características físicas de alto nivel, con una MLP en PyTorch.

- Dataset: `hls4ml_lhc_jets_hlf`, OpenML id 42468 — https://www.openml.org/d/42468
- DOI de los datos: https://doi.org/10.5281/zenodo.3602260 (licencia CC BY)
- Referencia del modelo base: Duarte et al. (2018), *Fast inference of deep neural
  networks in FPGAs for particle physics*, JINST 13 P07027.

## Instalación

```bash
conda create -n jets python=3.11 -y
conda activate jets
pip install -r requirements.txt
```

## Ejecución (desde la raíz del proyecto)

```bash
# 1) Descarga + EDA + creación de la partición 70/15/15 (semilla 42)
python -m scripts.eda

# 2) Modelo base (referencia hls4ml 64-32-32)
python -m src.train --config configs/baseline.json --seed 0
```

## Estructura

| Ruta | Contenido |
|---|---|
| `src/data.py` | descarga, partición estratificada, estandarización sin fuga |
| `src/model.py` | MLP configurable (profundidad, ancho, activación, dropout, normalización, inicialización) |
| `src/train.py` | entrenamiento, early stopping por F1-macro de validación, checkpoint, bitácora |
| `scripts/eda.py` | análisis exploratorio y datos para la ficha |
| `configs/` | una configuración JSON por experimento |
| `splits/` | índices de la partición (reproducibilidad) |
| `experiments/` | una carpeta por corrida + `bitacora.csv` |

## Protocolo

1. Partición 70/15/15 estratificada, semilla 42, guardada en `splits/split_seed42.npz`.
2. Estandarización ajustada solo con entrenamiento.
3. Selección de configuración solo con F1-macro de validación.
4. Configuración final repetida con ≥3 semillas (media ± desviación estándar).
5. Arquitectura y protocolo congelados antes de evaluar una sola vez en prueba.

## Licencia

Código bajo licencia MIT (ver `LICENSE`). Los datos no se incluyen: se
descargan de OpenML y conservan su licencia CC BY de origen.

## Declaración de asistencia de IA

Partes del código y la documentación se desarrollaron con asistencia de Claude
(Anthropic). Todos los experimentos y métricas reportados se obtuvieron
ejecutando este código.
