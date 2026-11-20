"""Summary table and figures from results/benchmark.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
COL = {"supervised": "black", "IAD": "#888888", "CATHODE": "#2a6fdb", "CWoLa": "#e07b39"}


def main():
    r = pd.DataFrame(json.loads((ROOT / "results/benchmark.json").read_text()))
    t = r.groupby(["method", "n_sig"]).agg(max_sic=("max_sic", "mean"), sd=("max_sic", "std"),
                                           s_sr=("sr_signal", "mean"), b_sr=("sr_background", "mean"),
                                           naive=("naive_significance", "mean"),
                                           boosted=("boosted_significance", "mean")).reset_index()
    t.round(3).to_csv(ROOT / "results/summary.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for m in ["IAD", "CATHODE", "CWoLa"]:
        g = t[(t.method == m) & (t.n_sig > 0)]
        sb = g.s_sr / g.b_sr * 100
        ax[0].errorbar(sb, g.max_sic, yerr=g.sd, marker="o", capsize=2, color=COL[m], label=m)
        ax[1].plot(g.naive, g.boosted, marker="o", color=COL[m], label=m)
    sup = t[t.method == "supervised"]
    ax[0].axhline(sup.max_sic.iloc[0], color="black", ls="--", lw=0.8, label="fully supervised")
    ax[0].set_xscale("log"); ax[0].set_xlabel("signal / background in the signal region (%)")
    ax[0].set_ylabel("max significance improvement (SIC)"); ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title("Anomaly detectors need a minimum amount of signal")
    x = np.linspace(0, 6.5, 10)
    ax[1].plot(x, x, color="black", lw=0.6, ls=":", label="no cut")
    ax[1].set_xlabel("significance before any cut (S / sqrt(B))"); ax[1].set_ylabel("significance after best cut")
    ax[1].set_title("A 3 sigma excess becomes an 18 to 35 sigma one"); ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "01_sic_vs_signal.png"); plt.close(fig)

    s = r[(r.n_sig == 1000) & (r.seed == 0)]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    grid = np.linspace(0.01, 1, 200)
    for _, row in s.iterrows():
        ax.plot(grid, row.sic_curve, color=COL[row.method], label=row.method)
    sup = r[r.method == "supervised"].iloc[0]
    ax.plot(grid, sup.sic_curve, color="black", ls="--", label="supervised (2000 injected)")
    ax.set_xlabel("signal efficiency"); ax.set_ylabel("SIC"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("1,000 injected signal events (S/B 1.2% in the signal region)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "02_sic_curves.png"); plt.close(fig)
    print(t.round(2).to_string())


if __name__ == "__main__":
    main()
