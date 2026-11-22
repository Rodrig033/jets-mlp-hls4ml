"""
Repite variantes de una etapa con varias semillas (misma partición) y reporta
media ± desviación estándar del F1-macro de validación.

Uso:
    python -m scripts.semillas configs/etapa2_capacidad.json \
        --variantes e2_64-32-32_control e2_256-128-64 --semillas 0 1 2
La semilla 0 se reutiliza si ya existe la corrida (no se repite).
"""
import argparse
import json

import numpy as np

from src.data import ROOT
from src.train import run

EXP = ROOT / "experiments"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa")
    ap.add_argument("--variantes", nargs="+", required=True)
    ap.add_argument("--semillas", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()

    spec = json.load(open(args.etapa))
    base = json.load(open(spec["base"]))
    variantes = {v["name"]: v for v in spec["variantes"]}
    tabla = []
    for nombre in args.variantes:
        cfg = {**base, **variantes[nombre]}
        f1s, accs = [], []
        for s in args.semillas:
            previo = EXP / f"{nombre}_s{s}" / "metrics.json"
            if previo.exists():
                m = json.load(open(previo))
                print(f"[reutilizada] {nombre}_s{s}: F1m {m['val_f1_macro']:.4f}")
            else:
                m = run(cfg, s)
            f1s.append(m["val_f1_macro"]); accs.append(m["val_acc"])
        tabla.append((nombre, np.mean(f1s), np.std(f1s, ddof=1), np.mean(accs),
                      np.std(accs, ddof=1), f1s))

    print(f"\n===== Media ± desv. est. (ddof=1) con semillas {args.semillas} =====")
    print(f"{'variante':28s} {'F1m':>17s} {'acc':>17s}   valores F1m")
    for n, mf, sf, ma, sa, vals in sorted(tabla, key=lambda t: -t[1]):
        print(f"{n:28s} {mf:.4f} ± {sf:.4f}   {ma:.4f} ± {sa:.4f}   "
              + " ".join(f"{v:.4f}" for v in vals))


if __name__ == "__main__":
    main()
