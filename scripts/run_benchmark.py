"""Run all four detectors at several injected signal levels and seeds, and save SIC results."""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lhco.data import FEATURES, build_data, in_sr, load, make_splits
from lhco.methods import cathode, cwola, iad, max_sic, sic_curve, supervised

import torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"
INJECT = [0, 250, 500, 1000, 2000]
SEEDS = [0, 1, 2]


def main():
    df = load(ROOT / "data" / "raw" / "events_anomalydetection_v2.features.h5")
    splits = make_splits(df)
    test = __import__("pandas").concat([splits["bkg_test"], splits["sig_test"]])
    test = test[in_sr(test.mjj)]
    X_test, y_test = test[FEATURES].values, test.label.values
    out_path = ROOT / "results" / "benchmark.json"
    res = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {(r["method"], r["n_sig"], r["seed"]) for r in res}
    curves = {}
    for n_sig in INJECT:
        for seed in SEEDS:
            data = build_data(splits, n_sig, seed)
            sr = data[in_sr(data.mjj)]
            n_s = int(sr.label.sum()); n_b = int(len(sr) - n_s)
            for name, fn in [("supervised", supervised), ("IAD", iad), ("CWoLa", cwola), ("CATHODE", cathode)]:
                if (name, n_sig, seed) in done or (name == "supervised" and n_sig != INJECT[-1]):
                    continue
                t = time.time()
                s = fn(data, splits, X_test, seed, device=DEV) if name == "CATHODE" else fn(data, splits, X_test, seed)
                ms = max_sic(s, y_test)
                g, sic, _, _ = sic_curve(s, y_test)
                res.append({"method": name, "n_sig": n_sig, "seed": seed, "sr_signal": n_s, "sr_background": n_b,
                            "naive_significance": n_s / np.sqrt(n_b), "max_sic": ms,
                            "boosted_significance": ms * n_s / np.sqrt(n_b), "seconds": time.time() - t,
                            "sic_curve": sic.tolist()})
                print(f"{name:10s} n_sig {n_sig:5d} seed {seed} S/sqrtB {n_s / np.sqrt(n_b):.2f} "
                      f"maxSIC {ms:.2f} ({time.time() - t:.0f}s)", flush=True)
                out_path.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
