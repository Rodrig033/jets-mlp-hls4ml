"""
Exporta los datos REALES que usa la presentación interactiva a un JSON pequeño:
  - curvas de entrenamiento (history.csv) del modelo base y del final (semilla 0)
  - curvas ROC uno-contra-el-resto (prueba, semilla 0), reducidas a una malla de FPR
  - matrices de confusión en prueba (semilla 0)
  - histogramas por clase de mass_mmdt y multiplicity (conjunto de ENTRENAMIENTO)
  - una selección de jets reales del conjunto de prueba con sus 16 observables y
    las probabilidades de ambos modelos (casos fáciles, ambiguos y errores)

Es un análisis posterior al congelamiento: no entrena ni selecciona nada.
Uso:  python -m scripts.exportar_presentacion --congelado
Salida: docs/presentacion_datos.json
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from scripts.roc_auc import probabilidades
from src.data import ROOT, encode, get_split, load_dataframe
from src.train import pick_device

EXP = ROOT / "experiments"
BASE, FINAL = "baseline_hls4ml", "e5_combo"
SALIDA = ROOT / "docs" / "presentacion_datos.json"


def historia(nombre):
    h = pd.read_csv(EXP / f"{nombre}_s0" / "history.csv")
    return {c: [round(float(v), 5) for v in h[c]] for c in
            ["epoch", "train_loss", "val_loss", "val_f1_macro", "val_acc"]}


def roc_reducida(P, y, clases):
    malla = np.unique(np.concatenate([np.logspace(-3.5, 0, 160), np.linspace(0, 1, 41)]))
    malla = malla[malla > 0]
    out = {"fpr": [round(float(f), 6) for f in malla], "tpr": {}}
    for k, c in enumerate(clases):
        fpr, tpr, _ = roc_curve((y == k).astype(int), P[:, k])
        out["tpr"][c] = [round(float(t), 4) for t in np.interp(malla, fpr, tpr)]
    return out


def matriz(nombre):
    return np.loadtxt(EXP / "prueba_final" / f"matriz_{nombre}_s0.csv",
                      delimiter=",", dtype=int).tolist()


def histogramas(df, clases, columna, bins):
    tr, _, _ = get_split(encode(df)[1], seed=42)
    sub = df.iloc[tr]
    out = {"bordes": [round(float(b), 3) for b in bins], "conteos": {}}
    for c in clases:
        v = sub.loc[sub["class"] == c, columna].to_numpy()
        h, _ = np.histogram(v, bins=bins)
        out["conteos"][c] = (h / h.sum()).round(5).tolist()  # fracción por bin
    return out


def seleccionar_jets(Pf, y, rng, por_clase=8):
    """Por clase: 4 aciertos seguros, 2 ambiguos (prob. máx. < 0.5) y 2 errores."""
    pred, conf = Pf.argmax(1), Pf.max(1)
    elegidos = []
    for k in range(Pf.shape[1]):
        grupos = {
            "facil": np.where((y == k) & (pred == k) & (conf > 0.8))[0],
            "ambiguo": np.where((y == k) & (conf < 0.5))[0],
            "error": np.where((y == k) & (pred != k) & (conf > 0.5))[0],
        }
        for tipo, n in (("facil", 4), ("ambiguo", 2), ("error", 2)):
            idx = grupos[tipo]
            if len(idx):
                for i in rng.choice(idx, size=min(n, len(idx)), replace=False):
                    elegidos.append((int(i), tipo))
    return elegidos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--congelado", action="store_true")
    args = ap.parse_args()
    if not args.congelado:
        raise SystemExit("Usa --congelado: este análisis lee el conjunto de prueba.")

    device = pick_device("auto")
    Pb, y, clases = probabilidades(BASE, 0, device)
    Pf, y2, _ = probabilidades(FINAL, 0, device)
    assert (y == y2).all()

    df = load_dataframe()
    X, y_all, feats, clases_all = encode(df)
    assert clases_all == clases
    _, _, te = get_split(y_all, seed=42)
    X_test = X[te]  # mismo orden que el X_test de prepare()

    rng = np.random.default_rng(2026)
    jets = []
    for i, tipo in seleccionar_jets(Pf, y, rng):
        jets.append({
            "tipo": tipo, "clase": clases[y[i]], "indice_prueba": i,
            "x": {f: round(float(v), 5) for f, v in zip(feats, X_test[i])},
            "p_final": [round(float(p), 4) for p in Pf[i]],
            "p_base": [round(float(p), 4) for p in Pb[i]],
        })
    rng.shuffle(jets)

    datos = {
        "clases": clases, "features": feats,
        "historia": {"base": historia(BASE), "final": historia(FINAL)},
        "roc": {"base": roc_reducida(Pb, y, clases), "final": roc_reducida(Pf, y, clases)},
        "matriz": {"base": matriz(BASE), "final": matriz(FINAL)},
        "hist_masa": histogramas(df, clases, "mass_mmdt", np.linspace(0, 300, 61)),
        "hist_mult": histogramas(df, clases, "multiplicity", np.arange(0, 152, 4)),
        "jets": jets,
        "nota": "Datos reales: prueba = semilla 0 de cada modelo; histogramas = entrenamiento. "
                "Los jets son una selección de casos fáciles, ambiguos y errores "
                "(no representativa de la tasa global).",
    }
    SALIDA.write_text(json.dumps(datos, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Guardado {SALIDA} ({SALIDA.stat().st_size / 1024:.0f} KB) · {len(jets)} jets")


if __name__ == "__main__":
    main()
