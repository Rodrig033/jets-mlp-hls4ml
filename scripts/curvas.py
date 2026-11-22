import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data import ROOT

EXP = ROOT / "experiments"
FIG = ROOT / "docs" / "fig"


def main(nombres):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for n in nombres:
        h = pd.read_csv(EXP / n / "history.csv")
        best = h.loc[h["val_f1_macro"].idxmax()]
        gap = best["val_loss"] - best["train_loss"]
        print(f"{n}: mejor época {int(best['epoch'])} | train_loss {best['train_loss']:.4f} | "
              f"val_loss {best['val_loss']:.4f} | brecha {gap:+.4f} | F1m {best['val_f1_macro']:.4f}")
        a1.plot(h["epoch"], h["train_loss"], label=f"{n} train")
        a1.plot(h["epoch"], h["val_loss"], "--", label=f"{n} val")
        a2.plot(h["epoch"], h["val_f1_macro"], label=n)
    a1.set_xlabel("época"); a1.set_ylabel("entropía cruzada"); a1.set_title("Pérdida")
    a2.set_xlabel("época"); a2.set_ylabel("F1-macro (validación)"); a2.set_title("F1-macro")
    a1.legend(fontsize=7); a2.legend(fontsize=7)
    fig.tight_layout()
    out = FIG / f"curvas_{'_vs_'.join(nombres)[:80]}.png"
    fig.savefig(out, dpi=200)
    print(f"\nFigura: {out}")
    print("Lectura: brecha ~0 y pérdidas altas -> subajuste (falta capacidad); "
          "val_loss subiendo mientras train baja -> sobreajuste (falta regularización).")


if __name__ == "__main__":
    main(sys.argv[1:])
