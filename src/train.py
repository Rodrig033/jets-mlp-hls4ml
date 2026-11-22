"""
Entrenamiento de una configuración. Uso:
    python -m src.train --config configs/baseline.json
    python -m src.train --config configs/baseline.json --seed 1

Cada corrida genera experiments/<nombre>_s<seed>/ con:
  config.json, history.csv (curvas), metrics.json, best.pt (checkpoint)
y agrega una fila a experiments/bitacora.csv.

IMPORTANTE: este script NUNCA carga el conjunto de prueba. La selección del
modelo se hace solo con validación (F1-macro), como pide la consigna.
"""
import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from src.data import ROOT, prepare
from src.model import MLP, count_params

EXP_DIR = ROOT / "experiments"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(pref="auto"):
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def hardware_info(device):
    return {
        "device": str(device),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }


@torch.no_grad()
def predict(model, X, batch=8192):
    model.eval()
    preds = []
    for i in range(0, len(X), batch):
        logits = model(X[i:i + batch])
        preds.append(logits.argmax(1))
    return torch.cat(preds).cpu().numpy()


@torch.no_grad()
def eval_loss(model, X, y, loss_fn, batch=8192):
    model.eval()
    total = 0.0
    for i in range(0, len(X), batch):
        total += loss_fn(model(X[i:i + batch]), y[i:i + batch]).item() * len(X[i:i + batch])
    return total / len(X)


def append_row(path, row):
    """Agrega una fila a la bitácora; si aparecen columnas nuevas, reescribe
    el encabezado conservando todas las filas anteriores (celdas vacías)."""
    rows, fields = [], list(row.keys())
    if path.exists():
        with open(path, newline="") as f:
            r = csv.DictReader(f)
            old = r.fieldnames or []
            rows = list(r)
        fields = old + [k for k in row if k not in old]
    rows.append(row)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


def build_optimizer(cfg, params):
    name = cfg.get("optimizer", "adam").lower()
    lr, wd = cfg["lr"], cfg.get("weight_decay", 0.0)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)
    raise ValueError(name)


def build_scheduler(cfg, opt, steps_per_epoch):
    name = cfg.get("scheduler", "none")
    if name == "none":
        return None, "epoch"
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"]), "epoch"
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3), "plateau"
    if name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=cfg["lr"], epochs=cfg["epochs"], steps_per_epoch=steps_per_epoch), "step"
    raise ValueError(name)


def run(cfg, seed):
    set_seed(seed)
    device = pick_device(cfg.get("device", "auto"))
    d = prepare(split_seed=cfg.get("split_seed", 42),  # sin prueba
                transform=cfg.get("transform", "standard"),
                drop_features=tuple(cfg.get("drop_features", [])))

    Xtr = torch.tensor(d["X_train"], device=device)
    ytr = torch.tensor(d["y_train"], device=device)
    Xva = torch.tensor(d["X_val"], device=device)
    yva = torch.tensor(d["y_val"], device=device)
    y_val_np = d["y_val"]

    model = MLP(
        n_in=Xtr.shape[1], n_out=len(d["class_names"]),
        hidden=tuple(cfg["hidden"]), activation=cfg.get("activation", "relu"),
        dropout=cfg.get("dropout", 0.0), norm=cfg.get("norm"), init=cfg.get("init", "default"),
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))
    opt = build_optimizer(cfg, model.parameters())
    bs = cfg["batch_size"]
    steps = int(np.ceil(len(Xtr) / bs))
    sched, sched_mode = build_scheduler(cfg, opt, steps)

    name = f"{cfg['name']}_s{seed}"
    out = EXP_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    json.dump({**cfg, "seed": seed}, open(out / "config.json", "w"), indent=2)

    history, best_f1, best_epoch, wait = [], -1.0, 0, 0
    patience = cfg.get("patience", 10)
    g = torch.Generator(device="cpu").manual_seed(seed)  # orden de minilotes reproducible
    t0 = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g).to(device)
        run_loss = 0.0
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(Xtr[b]), ytr[b])
            loss.backward()
            opt.step()
            if sched_mode == "step":
                sched.step()
            run_loss += loss.item() * len(b)
        train_loss = run_loss / len(Xtr)

        val_loss = eval_loss(model, Xva, yva, loss_fn)
        pred = predict(model, Xva)
        val_f1 = f1_score(y_val_np, pred, average="macro")
        val_acc = accuracy_score(y_val_np, pred)
        lr_now = opt.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_f1_macro": val_f1, "val_acc": val_acc, "lr": lr_now})
        print(f"[{name}] ep {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} "
              f"| F1m {val_f1:.4f} | acc {val_acc:.4f} | lr {lr_now:.2e}")

        if sched_mode == "epoch" and sched is not None:
            sched.step()
        elif sched_mode == "plateau":
            sched.step(val_f1)

        # Early stopping y checkpoint por F1-macro de validación.
        if val_f1 > best_f1:
            best_f1, best_epoch, wait = val_f1, epoch, 0
            torch.save({"model_state": model.state_dict(), "config": cfg, "seed": seed,
                        "class_names": d["class_names"], "feature_names": d["feature_names"],
                        "scaler_mean": d["scaler_mean"], "scaler_std": d["scaler_std"]},
                       out / "best.pt")
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping en época {epoch} (mejor: {best_epoch})")
                break

    elapsed = time.time() - t0
    with open(out / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys())
        w.writeheader()
        w.writerows(history)

    best = history[best_epoch - 1]
    metrics = {"name": cfg["name"], "seed": seed, "best_epoch": best_epoch,
               "epochs_run": len(history), "val_f1_macro": best["val_f1_macro"],
               "val_acc": best["val_acc"], "val_loss": best["val_loss"],
               "n_params": count_params(model), "train_time_s": round(elapsed, 1),
               **hardware_info(device)}
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2)

    # Bitácora global: una fila por corrida.
    log = EXP_DIR / "bitacora.csv"
    row = {"name": cfg["name"], "seed": seed, "hidden": "-".join(map(str, cfg["hidden"])),
           "activation": cfg.get("activation", "relu"), "dropout": cfg.get("dropout", 0.0),
           "norm": cfg.get("norm"), "optimizer": cfg.get("optimizer", "adam"),
           "lr": cfg["lr"], "batch_size": bs, "weight_decay": cfg.get("weight_decay", 0.0),
           "scheduler": cfg.get("scheduler", "none"), "best_epoch": best_epoch,
           "val_f1_macro": round(best["val_f1_macro"], 5), "val_acc": round(best["val_acc"], 5),
           "n_params": metrics["n_params"], "train_time_s": metrics["train_time_s"],
           "device": str(device), "nota": cfg.get("nota", ""),
           "transform": cfg.get("transform", "standard"),
           "drop_features": "|".join(cfg.get("drop_features", []))}
    append_row(log, row)

    print(f"\nListo: {name} | F1-macro val {best['val_f1_macro']:.4f} | "
          f"acc val {best['val_acc']:.4f} | {metrics['n_params']} params | {elapsed:.0f}s")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(json.load(open(args.config)), args.seed)
