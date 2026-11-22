"""
EVALUACIÓN FINAL EN PRUEBA. Debe ejecutarse UNA SOLA VEZ, después de congelar
la configuración final (git tag v1.0-congelado). Ninguna decisión puede
tomarse con lo que imprime este script.

Uso:
    python -m scripts.evaluate_test --congelado \
        --modelo baseline_hls4ml --modelo e5_combo --semillas 0 1 2

Para cada modelo y semilla carga experiments/<modelo>_s<semilla>/best.pt,
reconstruye el preprocesamiento con su configuración (ajustado solo con
entrenamiento) y evalúa el conjunto de prueba completo (124,500 jets).
Salidas en experiments/prueba_final/:
  resultados.json, resumen.csv, reporte_<modelo>_s<semilla>.txt,
  matriz_<modelo>_s<semilla>.csv y docs/fig/matriz_<modelo>.png (semilla 0)
"""
import argparse
import csv
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

from src.data import ROOT, prepare
from src.model import MLP, count_params
from src.train import hardware_info, pick_device

EXP = ROOT / "experiments"
OUT = EXP / "prueba_final"
FIG = ROOT / "docs" / "fig"


def evaluar(nombre, seed, device):
    ck_path = EXP / f"{nombre}_s{seed}" / "best.pt"
    # weights_only=False: el checkpoint es propio y contiene arreglos numpy/listas.
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
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
    y = d["y_test"]
    with torch.no_grad():
        model(X[:4096])  # calentamiento
        t0 = time.perf_counter()
        pred = torch.cat([model(X[i:i + 8192]).argmax(1) for i in range(0, len(X), 8192)])
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        t_inf = time.perf_counter() - t0
    pred = pred.cpu().numpy()

    clases = d["class_names"]
    cm = confusion_matrix(y, pred)
    res = {
        "modelo": nombre, "seed": seed, "n_test": int(len(y)),
        "f1_macro": f1_score(y, pred, average="macro"),
        "accuracy": accuracy_score(y, pred),
        "precision_macro": precision_score(y, pred, average="macro"),
        "recall_macro": recall_score(y, pred, average="macro"),
        "f1_por_clase": dict(zip(clases, f1_score(y, pred, average=None).round(4).tolist())),
        "n_params": count_params(model),
        "inferencia_total_s": round(t_inf, 4),
        "inferencia_us_por_jet": round(t_inf / len(y) * 1e6, 3),
        "matriz_confusion": cm.tolist(), "clases": clases,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"reporte_{nombre}_s{seed}.txt").write_text(
        classification_report(y, pred, target_names=clases, digits=4), encoding="utf-8")
    np.savetxt(OUT / f"matriz_{nombre}_s{seed}.csv", cm, fmt="%d", delimiter=",",
               header=",".join(clases))
    return res


def graficar_matriz(res, nombre):
    cm = np.array(res["matriz_confusion"], dtype=float)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    clases = res["clases"]
    fig, ax = plt.subplots(figsize=(5, 4.3))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(clases)):
        for j in range(len(clases)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xticks(range(len(clases))); ax.set_xticklabels(clases)
    ax.set_yticks(range(len(clases))); ax.set_yticklabels(clases)
    ax.set_xlabel("Predicción"); ax.set_ylabel("Clase real")
    ax.set_title(f"{nombre} (prueba, normalizada por fila)")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"matriz_{nombre}.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", action="append", required=True)
    ap.add_argument("--semillas", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--congelado", action="store_true",
                    help="confirma que la configuración final ya está congelada")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    if not args.congelado:
        raise SystemExit("Abortado: agrega --congelado solo cuando la configuración final "
                         "esté congelada (git tag). La prueba se evalúa una sola vez.")

    device = pick_device(args.device)
    todos = []
    for nombre in args.modelo:
        for s in args.semillas:
            r = evaluar(nombre, s, device)
            todos.append(r)
            print(f"{nombre}_s{s}: F1m {r['f1_macro']:.4f} | acc {r['accuracy']:.4f} | "
                  f"P {r['precision_macro']:.4f} | R {r['recall_macro']:.4f} | "
                  f"{r['inferencia_us_por_jet']} µs/jet")
            if s == args.semillas[0]:
                graficar_matriz(r, nombre)

    print("\n===== PRUEBA: media ± desv. est. (ddof=1) =====")
    filas = []
    for nombre in args.modelo:
        rs = [r for r in todos if r["modelo"] == nombre]
        f1 = np.array([r["f1_macro"] for r in rs]); acc = np.array([r["accuracy"] for r in rs])
        sd = lambda a: a.std(ddof=1) if len(a) > 1 else 0.0
        fila = {"modelo": nombre, "semillas": len(rs),
                "f1_macro_media": round(f1.mean(), 4), "f1_macro_std": round(sd(f1), 4),
                "acc_media": round(acc.mean(), 4), "acc_std": round(sd(acc), 4),
                "n_params": rs[0]["n_params"]}
        for c in rs[0]["clases"]:
            fila[f"f1_{c}"] = round(np.mean([r["f1_por_clase"][c] for r in rs]), 4)
        filas.append(fila)
        print(f"{nombre:20s} F1m {fila['f1_macro_media']:.4f} ± {fila['f1_macro_std']:.4f} | "
              f"acc {fila['acc_media']:.4f} ± {fila['acc_std']:.4f} | "
              + " ".join(f"{c}={fila[f'f1_{c}']:.3f}" for c in rs[0]["clases"]))

    json.dump({"hardware": hardware_info(device), "corridas": todos, "resumen": filas},
              open(OUT / "resultados.json", "w"), indent=2, ensure_ascii=False)
    with open(OUT / "resumen.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=filas[0].keys())
        w.writeheader(); w.writerows(filas)
    print(f"\nResultados en {OUT} · matrices en {FIG}")


if __name__ == "__main__":
    main()
