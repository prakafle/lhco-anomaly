"""Load the LHCO R&D high-level features and build the standard 5 features used in the CATHODE paper."""
import numpy as np
import pandas as pd

SR = (3.3, 3.7)        # signal region in dijet mass, TeV
FEATURES = ["mj1", "dmj", "tau21_1", "tau21_2"]


def load(path) -> pd.DataFrame:
    df = pd.read_hdf(path)
    out = pd.DataFrame()
    for j in (1, 2):
        p2 = df[f"pxj{j}"] ** 2 + df[f"pyj{j}"] ** 2 + df[f"pzj{j}"] ** 2
        df[f"ej{j}"] = np.sqrt(p2 + df[f"mj{j}"] ** 2)
    px, py, pz, e = (df[f"{c}j1"] + df[f"{c}j2"] for c in ("px", "py", "pz", "e"))
    out["mjj"] = np.sqrt(np.clip(e ** 2 - px ** 2 - py ** 2 - pz ** 2, 0, None)) / 1000.0
    # Order jets by mass: j1 is the lighter jet.
    light = df["mj1"] <= df["mj2"]
    m_lo = np.where(light, df["mj1"], df["mj2"]) / 1000.0
    m_hi = np.where(light, df["mj2"], df["mj1"]) / 1000.0
    t21_a = df["tau2j1"] / df["tau1j1"].replace(0, np.nan)
    t21_b = df["tau2j2"] / df["tau1j2"].replace(0, np.nan)
    out["mj1"] = m_lo
    out["dmj"] = m_hi - m_lo
    out["tau21_1"] = np.where(light, t21_a, t21_b)
    out["tau21_2"] = np.where(light, t21_b, t21_a)
    out["label"] = df["label"].astype(int)
    return out.dropna().reset_index(drop=True)


def in_sr(mjj):
    return (mjj > SR[0]) & (mjj < SR[1])


def make_splits(df: pd.DataFrame, seed: int = 0):
    """Fixed split of the sample.
    - 'data' background: 500k events, plus injected signal at a chosen level.
    - 'template' background: 300k events, used only by the idealized anomaly detector (perfect background model).
    - test: the remaining 200k background and a disjoint half of the signal, signal region only.
    """
    rng = np.random.default_rng(seed)
    bkg = df[df.label == 0].sample(frac=1, random_state=seed).reset_index(drop=True)
    sig = df[df.label == 1].sample(frac=1, random_state=seed).reset_index(drop=True)
    half = len(sig) // 2
    return {
        "bkg_data": bkg.iloc[:500_000], "bkg_template": bkg.iloc[500_000:800_000], "bkg_test": bkg.iloc[800_000:],
        "sig_pool": sig.iloc[:half], "sig_test": sig.iloc[half:],
    }


def build_data(splits, n_sig: int, seed: int):
    """'Observed' data: 500k background plus n_sig signal events (counted over the full mass range)."""
    s = splits["sig_pool"].sample(n=n_sig, random_state=seed) if n_sig else splits["sig_pool"].iloc[:0]
    return pd.concat([splits["bkg_data"], s]).sample(frac=1, random_state=seed).reset_index(drop=True)
