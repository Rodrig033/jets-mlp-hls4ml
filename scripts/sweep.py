import argparse
import json

from src.train import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--solo", default=None, help="correr solo la variante con este nombre")
    args = ap.parse_args()

    spec = json.load(open(args.etapa))
    base = json.load(open(spec["base"]))
    resultados = []
    for var in spec["variantes"]:
        if args.solo and var["name"] != args.solo:
            continue
        cfg = {**base, **var}
        print(f"\n===== {cfg['name']} =====")
        m = run(cfg, args.seed)
        resultados.append((cfg["name"], m["val_f1_macro"], m["val_acc"],
                           m["n_params"], m["train_time_s"], m["best_epoch"]))

    print(f"\n===== Resumen de {args.etapa} (ordenado por F1-macro de validación) =====")
    print(f"{'nombre':32s} {'F1m':>7s} {'acc':>7s} {'params':>9s} {'seg':>7s} {'ep*':>4s}")
    for r in sorted(resultados, key=lambda t: -t[1]):
        print(f"{r[0]:32s} {r[1]:7.4f} {r[2]:7.4f} {r[3]:9,d} {r[4]:7.1f} {r[5]:4d}")


if __name__ == "__main__":
    main()
