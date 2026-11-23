"""
Curvas ROC y AUC (uno contra el resto) en el conjunto de PRUEBA para modelos ya
congelados. Es un análisis posterior a la evaluación final: no entrena ni
selecciona nada, solo calcula métricas adicionales sobre los mismos checkpoints
y el mismo conjunto de prueba.

Métricas:
  - AUC por clase (uno contra el resto) y AUC macro (promedio de las 5).
  - Eficiencia de señal a tasa fija de fondo: fracción de jets de la clase que
    se conservan cuando solo se acepta el 10 % y el 1 % de los jets de las demás
    clases (forma estándar de evaluar etiquetadores en física de partículas).

Uso:
    python -m scripts.roc_auc --congelado \
        --modelo baseline_hls4ml --modelo e5_combo --semillas 0 1 2
Salidas: experiments/prueba_final/roc_auc.json, roc_auc.csv y
docs/fig/roc_<modelo>.png, docs/fig/roc_comparacion.png
"""
import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from src.data import ROOT, prepare
from src.model import MLP
from src.train import pick_device

EXP = ROOT / "experiments"
OUT = EXP / "prueba_final"
FIG = ROOT / "docs" / "fig"
FPR_FIJOS = (0.10, 0.01)


def probabilidades(nombre, seed, device):
    """Carga el checkpoint y devuelve (probabilidades softmax, y_test, clases)."""
    ck = torch.load(EXP / f"{nombre}_s{seed}" / "best.pt", map_location="cpu",
                    weights_only=False)  # checkpoint propio (contiene numpy/listas)
    cfg = ck["config"]
    d = prepare(split_seed=cfg.get("split_seed", 42), include_test=True,
                transform=cfg.get("transform", "standard"),
                drop_features=tuple(cfg.get("drop_features", [])))
    assert d["feature_names"] == ck["feature_names"], "Las características no coinciden"
    model = MLP(n_in=len(d["feature_names"]), n_out=len(d["class_names"]),
                hidden=tuple(cfg["hidden"]), activation=cfg.get("activation", "relu"),
                dropout=cfg.get("dropout", 0.0), norm=cfg.get("norm"),
                init=cfg.get("init", "default")).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    X = torch.tensor(d["X_test"], device=device)
    with torch.no_grad():
        P = torch.cat([torch.softmax(model(X[i:i + 8192]), dim=1)
                       for i in range(0, len(X), 8192)]).cpu().numpy()
    return P, d["y_test"], d["class_names"]


def metricas_roc(P, y, clases):
    """AUC uno-contra-resto por clase, AUC macro y eficiencias a FPR fijo."""
    res = {"auc": {}, "eff": {}, "curvas": {}}
    for k, c in enumerate(clases):
        señal = (y == k).astype(int)
        fpr, tpr, _ = roc_curve(señal, P[:, k])
        res["auc"][c] = float(roc_auc_score(señal, P[:, k]))
        # roc_curve devuelve fpr creciente: se interpola TPR en el FPR pedido
        res["eff"][c] = {f"{f:g}": float(np.interp(f, fpr, tpr)) for f in FPR_FIJOS}
        res["curvas"][c] = (fpr, tpr)
    res["auc_macro"] = float(np.mean(list(res["auc"].values())))
    # Comprobación cruzada con la implementación multiclase de sklearn
    res["auc_macro_sklearn"] = float(roc_auc_score(y, P, multi_class="ovr", average="macro"))
    return res


def graficar(curvas_por_modelo, clases, archivo, titulo):
    """Estilo física: eficiencia de señal (x) vs. tasa de fondo aceptado (y, log)."""
    colores = dict(zip(clases, ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]))
    estilos = ["-", "--", ":"]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for (etiqueta, curvas, aucs), ls in zip(curvas_por_modelo, estilos):
        for c in clases:
            fpr, tpr = curvas[c]
            ax.plot(tpr, fpr, ls, color=colores[c], lw=1.4,
                    label=f"{c} {etiqueta} (AUC {aucs[c]:.3f})")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1); ax.set_xlim(0, 1)
    ax.set_xlabel("Eficiencia de señal (TPR)")
    ax.set_ylabel("Tasa de fondo aceptado (FPR)")
    ax.set_title(titulo)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=6.5, ncol=1 if len(curvas_por_modelo) == 1 else 2, loc="lower right")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / archivo, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", action="append", required=True)
    ap.add_argument("--semillas", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--congelado", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    if not args.congelado:
        raise SystemExit("Abortado: este análisis usa el conjunto de prueba. Agrega "
                         "--congelado solo sobre modelos ya congelados.")

    device = pick_device(args.device)
    corridas, para_grafica, clases = [], [], None
    for nombre in args.modelo:
        for s in args.semillas:
            P, y, clases = probabilidades(nombre, s, device)
            m = metricas_roc(P, y, clases)
            corridas.append({"modelo": nombre, "seed": s, "auc": m["auc"],
                             "auc_macro": m["auc_macro"],
                             "auc_macro_sklearn": m["auc_macro_sklearn"], "eff": m["eff"]})
            print(f"{nombre}_s{s}: AUC macro {m['auc_macro']:.4f} | "
                  + " ".join(f"{c}={m['auc'][c]:.4f}" for c in clases))
            if s == args.semillas[0]:
                graficar([("", m["curvas"], m["auc"])], clases, f"roc_{nombre}.png",
                         f"{nombre} (prueba, semilla {s})")
                para_grafica.append((nombre, m["curvas"], m["auc"]))

    if len(para_grafica) > 1:
        graficar(para_grafica, clases, "roc_comparacion.png",
                 "ROC uno contra el resto (prueba, semilla 0)")

    print("\n===== PRUEBA: media ± desv. est. (ddof=1) =====")
    filas = []
    sd = lambda a: float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
    for nombre in args.modelo:
        rs = [r for r in corridas if r["modelo"] == nombre]
        macro = [r["auc_macro"] for r in rs]
        fila = {"modelo": nombre, "auc_macro_media": round(float(np.mean(macro)), 4),
                "auc_macro_std": round(sd(macro), 4)}
        for c in clases:
            fila[f"auc_{c}"] = round(float(np.mean([r["auc"][c] for r in rs])), 4)
        for f in FPR_FIJOS:
            for c in clases:
                fila[f"eff_{c}@fpr{f:g}"] = round(
                    float(np.mean([r["eff"][c][f"{f:g}"] for r in rs])), 4)
        filas.append(fila)
        print(f"{nombre:18s} AUC macro {fila['auc_macro_media']:.4f} ± {fila['auc_macro_std']:.4f} | "
              + " ".join(f"{c}={fila[f'auc_{c}']:.4f}" for c in clases))
        for f in FPR_FIJOS:
            print(f"{'':18s} eficiencia con {f:.0%} de fondo: "
                  + " ".join(f"{c}={fila[f'eff_{c}@fpr{f:g}']:.3f}" for c in clases))

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"corridas": corridas, "resumen": filas}, open(OUT / "roc_auc.json", "w"),
              indent=2, ensure_ascii=False)
    with open(OUT / "roc_auc.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=filas[0].keys())
        w.writeheader(); w.writerows(filas)
    print(f"\nResultados en {OUT} · figuras en {FIG}")


if __name__ == "__main__":
    main()
