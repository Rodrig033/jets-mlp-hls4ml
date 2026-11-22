"""
Carga, partición y preprocesamiento del dataset hls4ml LHC Jets (OpenML id 42468).

Decisiones clave (para defender en clase):
- OpenML no prescribe una partición oficial y los jets son eventos simulados
  independientes (no hay pacientes, grupos ni orden temporal). Por eso, según
  la consigna, usamos 70/15/15 estratificado con semilla registrada.
- La estandarización (media 0, varianza 1) se AJUSTA SOLO con entrenamiento y
  se aplica a validación y prueba. Ajustarla con todo el dataset filtraría
  información de prueba al modelo (fuga de información).
- Las particiones se guardan como índices en disco: cualquier corrida futura
  (base, búsqueda, semillas, prueba final) usa exactamente las mismas filas.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

OPENML_ID = 42468
ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "jets_hlf.csv"
SPLIT_DIR = ROOT / "splits"
TARGET = "class"


def download(force: bool = False) -> Path:
    """Descarga el dataset de OpenML una sola vez y lo guarda como CSV local."""
    if DATA_FILE.exists() and not force:
        return DATA_FILE
    from sklearn.datasets import fetch_openml

    print(f"Descargando OpenML id={OPENML_ID} (puede tardar unos minutos)...")
    bunch = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="auto")
    df = bunch.frame
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_FILE, index=False)
    print(f"Guardado en {DATA_FILE} -> {df.shape}")
    return DATA_FILE


def load_dataframe() -> pd.DataFrame:
    if not DATA_FILE.exists():
        download()
    df = pd.read_csv(DATA_FILE)
    df[TARGET] = df[TARGET].astype(str)
    return df


def encode(df: pd.DataFrame):
    """Separa X / y y codifica las clases como enteros 0..K-1 (orden alfabético)."""
    feature_names = [c for c in df.columns if c != TARGET]
    X = df[feature_names].to_numpy(dtype=np.float32)
    class_names = sorted(df[TARGET].unique())
    mapping = {c: i for i, c in enumerate(class_names)}
    y = df[TARGET].map(mapping).to_numpy(dtype=np.int64)
    return X, y, feature_names, class_names


def make_split(y: np.ndarray, seed: int = 42, val: float = 0.15, test: float = 0.15):
    """Partición estratificada 70/15/15. Devuelve índices (no datos)."""
    idx = np.arange(len(y))
    idx_train, idx_tmp = train_test_split(
        idx, test_size=val + test, stratify=y, random_state=seed
    )
    # Del 30 % restante, la mitad es validación y la mitad prueba.
    idx_val, idx_test = train_test_split(
        idx_tmp, test_size=test / (val + test), stratify=y[idx_tmp], random_state=seed
    )
    return np.sort(idx_train), np.sort(idx_val), np.sort(idx_test)


def get_split(y: np.ndarray, seed: int = 42):
    """Carga la partición guardada; si no existe la crea y la guarda."""
    path = SPLIT_DIR / f"split_seed{seed}.npz"
    if path.exists():
        s = np.load(path)
        assert len(s["train"]) + len(s["val"]) + len(s["test"]) == len(y), \
            "La partición guardada no coincide con el tamaño del dataset."
        return s["train"], s["val"], s["test"]
    tr, va, te = make_split(y, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, train=tr, val=va, test=te)
    print(f"Partición nueva guardada en {path}")
    return tr, va, te


class Standardizer:
    """z = (x - media_train) / desv_train. Se ajusta SOLO con entrenamiento."""

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0  # evita división entre cero
        return self

    def transform(self, X):
        return ((X - self.mean_) / self.std_).astype(np.float32)


def prepare(split_seed: int = 42, include_test: bool = False):
    """
    Punto de entrada para entrenamiento. Por defecto NO devuelve prueba:
    la prueba solo se toca en la evaluación final (scripts/evaluate_test.py).
    """
    df = load_dataframe()
    X, y, feature_names, class_names = encode(df)
    tr, va, te = get_split(y, seed=split_seed)
    scaler = Standardizer().fit(X[tr])
    out = {
        "X_train": scaler.transform(X[tr]), "y_train": y[tr],
        "X_val": scaler.transform(X[va]), "y_val": y[va],
        "feature_names": feature_names, "class_names": class_names,
        "scaler_mean": scaler.mean_, "scaler_std": scaler.std_,
    }
    if include_test:
        out["X_test"] = scaler.transform(X[te])
        out["y_test"] = y[te]
    return out
