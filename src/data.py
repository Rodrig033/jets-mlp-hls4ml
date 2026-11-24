from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

OPENML_ID = 42468
ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "jets_hlf.csv"
SPLIT_DIR = ROOT / "splits"
TARGET = "class"


ARFF_URL = "https://api.openml.org/data/v1/download/21854360/hls4ml_lhc_jets_hlf.arff"
ARFF_FILE = ROOT / "data" / "hls4ml_lhc_jets_hlf.arff"
INFO_FILE = ROOT / "data" / "DATA_INFO.json"


def _hashes(path: Path):
    import hashlib
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            md5.update(chunk)
            sha.update(chunk)
    return md5.hexdigest(), sha.hexdigest()


def _download_arff():
    """Descarga directa del ARFF con barra de progreso simple."""
    import urllib.request
    ARFF_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = ARFF_FILE.with_suffix(".part")
    print(f"Descarga directa: {ARFF_URL}")
    with urllib.request.urlopen(ARFF_URL) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            msg = f"{done / 1e6:7.1f} MB" + (f" / {total / 1e6:.1f} MB" if total else "")
            print("\r  " + msg, end="", flush=True)
    print()
    tmp.rename(ARFF_FILE)


def _parse_arff(path: Path) -> pd.DataFrame:
    """Lector ARFF mínimo: atributos del encabezado + bloque @data como CSV."""
    cols = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s.lower().startswith("@attribute"):
                cols.append(s.split()[1].strip("'\""))
            elif s.lower().startswith("@data"):
                break
        df = pd.read_csv(f, header=None, names=cols, comment="%",
                         quotechar="'", skipinitialspace=True)
    return df


def _validate(df: pd.DataFrame):
    """Chequeos de integridad que sustituyen al md5 de OpenML."""
    assert TARGET in df.columns, f"No existe la columna '{TARGET}'"
    assert df.shape[1] == 17, f"Se esperaban 17 columnas, hay {df.shape[1]}"
    assert int(df.isna().sum().sum()) == 0, "Hay valores faltantes/no parseables"
    n_cls = df[TARGET].nunique()
    assert n_cls == 5, f"Se esperaban 5 clases, hay {n_cls}"


def download(force: bool = False) -> Path:
    if DATA_FILE.exists() and not force:
        return DATA_FILE
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    info = {"openml_id": OPENML_ID, "url": ARFF_URL,
            "fecha_descarga": pd.Timestamp.now().isoformat(timespec="seconds")}

    df = None
    if not ARFF_FILE.exists():
        try:
            from sklearn.datasets import fetch_openml
            print(f"Descargando OpenML id={OPENML_ID} con fetch_openml...")
            df = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="auto").frame
            info["metodo"] = "sklearn.fetch_openml (md5 verificado)"
        except ValueError as e:
            if "md5" not in str(e):
                raise
            print("\nAVISO: el md5 registrado en OpenML no coincide con el archivo servido.")
            print("Se usará descarga directa + validación estructural.\n")
            info["aviso_md5_openml"] = str(e).split("description:")[-1].strip()
            _download_arff()

    if df is None:
        df = _parse_arff(ARFF_FILE)
        md5, sha = _hashes(ARFF_FILE)
        info.update({"metodo": info.get("metodo", "ARFF local / descarga directa"),
                     "arff_md5": md5, "arff_sha256": sha})

    _validate(df)
    df[TARGET] = df[TARGET].astype(str).str.strip()
    info.update({"filas": int(len(df)), "columnas": int(df.shape[1]),
                 "clases": df[TARGET].value_counts().sort_index().to_dict()})
    df.to_csv(DATA_FILE, index=False)
    import json
    json.dump(info, open(INFO_FILE, "w"), indent=2, ensure_ascii=False)
    print(f"Guardado en {DATA_FILE} -> {df.shape}")
    print(f"Registro de integridad en {INFO_FILE}")
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


class YeoJohnson:
    """
    Transformación de potencia Yeo-Johnson + estandarización (sklearn).
    Reduce la asimetría de variables sesgadas. Se ajusta SOLO con entrenamiento.
    """

    def fit(self, X):
        from sklearn.preprocessing import PowerTransformer
        self.pt = PowerTransformer(method="yeo-johnson", standardize=True)
        # Ajustar sobre una submuestra fija de entrenamiento basta para estimar
        # los lambdas y evita que tarde minutos con 581k filas.
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X), size=min(100_000, len(X)), replace=False)
        self.pt.fit(X[idx].astype(np.float64))
        self.mean_ = self.std_ = None
        return self

    def transform(self, X):
        return self.pt.transform(X.astype(np.float64)).astype(np.float32)


def prepare(split_seed: int = 42, include_test: bool = False,
            transform: str = "standard", drop_features=()):
    """
    Punto de entrada para entrenamiento. Por defecto NO devuelve prueba:
    la prueba solo se toca en la evaluación final (scripts/evaluate_test.py).

    transform: "standard" (z-score) o "yeojohnson" (potencia + z-score).
    drop_features: nombres de columnas a excluir (p. ej. la D2 duplicada).
    """
    df = load_dataframe()
    if drop_features:
        df = df.drop(columns=list(drop_features))
    X, y, feature_names, class_names = encode(df)
    tr, va, te = get_split(y, seed=split_seed)
    scaler = (YeoJohnson() if transform == "yeojohnson" else Standardizer()).fit(X[tr])
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
