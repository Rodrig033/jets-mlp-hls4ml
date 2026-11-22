"""
Paso 1: descarga + análisis exploratorio + datos para la ficha del dataset.
Uso (desde la raíz del proyecto):  python -m scripts.eda
Genera docs/eda_resumen.json, docs/eda_resumen.txt y figuras en docs/fig/.
No usa PyTorch.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import ROOT, TARGET, encode, get_split, load_dataframe

OUT = ROOT / "docs"
FIG = OUT / "fig"
FIG.mkdir(parents=True, exist_ok=True)


def main():
    df = load_dataframe()
    X, y, feats, classes = encode(df)
    lines = []
    p = lines.append

    p(f"Muestras: {len(df):,}   Características: {len(feats)}   Clases: {len(classes)}")
    p(f"Duplicados exactos (filas completas): {df.duplicated().sum():,}")
    p(f"Valores faltantes totales: {int(df.isna().sum().sum())}")
    p("\nTipos de variables:")
    p(df.dtypes.value_counts().to_string())

    counts = df[TARGET].value_counts().sort_index()
    p("\nFrecuencia por clase:")
    for c, n in counts.items():
        p(f"  {c:>4}: {n:>8,}  ({n / len(df):.2%})")

    desc = df[feats].describe().T
    desc["skew"] = df[feats].skew()
    desc["n_unicos"] = df[feats].nunique()
    p("\nEstadísticos por característica:")
    p(desc.round(4).to_string())

    # Correlación entre características (redundancia)
    corr = df[feats].corr().abs()
    pairs = [(feats[i], feats[j], corr.iloc[i, j])
             for i in range(len(feats)) for j in range(i + 1, len(feats))]
    pairs.sort(key=lambda t: -t[2])
    p("\nPares de características más correlacionadas (|r|):")
    for a, b, r in pairs[:10]:
        p(f"  {a} ~ {b}: {r:.3f}")

    # Partición: se crea y guarda aquí por primera vez
    tr, va, te = get_split(y, seed=42)
    p(f"\nPartición 70/15/15 (semilla 42): train {len(tr):,} | val {len(va):,} | test {len(te):,}")
    for nombre, idx in [("train", tr), ("val", va), ("test", te)]:
        dist = np.bincount(y[idx], minlength=len(classes)) / len(idx)
        p(f"  {nombre:5s}: " + "  ".join(f"{c}={v:.3f}" for c, v in zip(classes, dist)))

    texto = "\n".join(lines)
    print(texto)
    (OUT / "eda_resumen.txt").write_text(texto, encoding="utf-8")
    json.dump({"n": len(df), "features": feats, "classes": classes,
               "counts": counts.to_dict(), "missing": int(df.isna().sum().sum()),
               "duplicates": int(df.duplicated().sum()),
               "split": {"train": len(tr), "val": len(va), "test": len(te)}},
              open(OUT / "eda_resumen.json", "w"), indent=2)

    # Figuras
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(counts.index, counts.values, color="#4C72B0")
    ax.set_ylabel("Jets"); ax.set_title("Distribución de clases")
    fig.tight_layout(); fig.savefig(FIG / "clases.png", dpi=200); plt.close(fig)

    n = len(feats); cols = 4; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3 * rows))
    sample = df.sample(min(100_000, len(df)), random_state=0)
    for ax, f in zip(axes.flat, feats):
        for c in classes:
            v = sample.loc[sample[TARGET] == c, f]
            lo, hi = np.percentile(sample[f], [0.5, 99.5])
            ax.hist(v.clip(lo, hi), bins=60, histtype="step", density=True, label=c)
        ax.set_title(f, fontsize=9)
    for ax in axes.flat[n:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "distribuciones_por_clase.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(df[feats].corr(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(feats, rotation=90, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(feats, fontsize=7)
    fig.colorbar(im); ax.set_title("Correlación de Pearson")
    fig.tight_layout(); fig.savefig(FIG / "correlacion.png", dpi=200); plt.close(fig)
    print(f"\nFiguras en {FIG}")


if __name__ == "__main__":
    main()
