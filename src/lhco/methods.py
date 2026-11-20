"""The four detectors.

All of them end the same way: a classifier separates SR data from some reference sample. If the reference is a
faithful picture of background, the classifier can only win by finding the signal (Neyman-Pearson via
likelihood ratio). They differ only in where the reference comes from:
  supervised  labelled signal vs background (an upper bound, not an anomaly detector)
  IAD         a perfect background sample in the SR (idealized, unavailable in real life)
  CWoLa       data in the neighbouring sidebands (no generative model; breaks if features correlate with mjj)
  CATHODE     samples from a flow p(x | mjj) trained in sidebands and interpolated into the SR
"""
import numpy as np
import torch
import zuko
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

from .data import FEATURES, SR, in_sr

SB_WIDTH = 0.2  # TeV either side of the SR used by CWoLa


def classifier_scores(X_a, X_b, X_test, n_models=5, seed=0):
    """Train an ensemble of boosted trees to tell sample a (label 1) from sample b (label 0).
    Returns the mean predicted score on X_test. Uses k-fold so each model sees a different 80%."""
    X = np.vstack([X_a, X_b]); y = np.r_[np.ones(len(X_a)), np.zeros(len(X_b))]
    w = np.where(y == 1, 0.5 / len(X_a), 0.5 / len(X_b)) * len(y)
    scores = np.zeros(len(X_test))
    skf = StratifiedKFold(n_splits=n_models, shuffle=True, random_state=seed)
    for i, (tr, va) in enumerate(skf.split(X, y)):
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                             early_stopping=True, validation_fraction=0.2, n_iter_no_change=10,
                                             random_state=seed + i)
        clf.fit(X[tr], y[tr], sample_weight=w[tr])
        scores += clf.predict_proba(X_test)[:, 1] / n_models
    return scores


def sr(df):
    return df[in_sr(df.mjj)]


def supervised(data, splits, X_test, seed):
    s = sr(splits["sig_pool"])[FEATURES].values
    b = sr(splits["bkg_template"])[FEATURES].values
    return classifier_scores(s, b, X_test, seed=seed)


def iad(data, splits, X_test, seed):
    return classifier_scores(sr(data)[FEATURES].values, sr(splits["bkg_template"])[FEATURES].values, X_test, seed=seed)


def cwola(data, splits, X_test, seed):
    lo, hi = SR
    sb = data[((data.mjj > lo - SB_WIDTH) & (data.mjj < lo)) | ((data.mjj > hi) & (data.mjj < hi + SB_WIDTH))]
    return classifier_scores(sr(data)[FEATURES].values, sb[FEATURES].values, X_test, seed=seed)


class Standardizer:
    def __init__(self, X):
        self.mu, self.sd = X.mean(0), X.std(0)

    def fwd(self, X):
        return (X - self.mu) / self.sd

    def inv(self, Z):
        return Z * self.sd + self.mu


def train_flow(X, m, epochs=60, batch=1024, lr=1e-3, seed=0, device="cpu"):
    """Conditional neural spline flow p(x | mjj) on sideband data. Returns (flow, scalers, val-loss history)."""
    torch.manual_seed(seed)
    # logit-like transform keeps tau21 inside (0, 1) and masses positive.
    sx = Standardizer(X); sm = Standardizer(m[:, None])
    Xt = torch.tensor(sx.fwd(X), dtype=torch.float32); Mt = torch.tensor(sm.fwd(m[:, None]), dtype=torch.float32)
    n = len(Xt); perm = torch.randperm(n); nv = n // 5
    va, tr = perm[:nv], perm[nv:]
    flow = zuko.flows.NSF(features=X.shape[1], context=1, transforms=6, hidden_features=[128, 128]).to(device)
    opt = torch.optim.AdamW(flow.parameters(), lr=lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, best_state, hist = 1e9, None, []
    for ep in range(epochs):
        flow.train()
        idx = tr[torch.randperm(len(tr))]
        for i in range(0, len(idx), batch):
            b = idx[i:i + batch]
            loss = -flow(Mt[b].to(device)).log_prob(Xt[b].to(device)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        flow.eval()
        with torch.no_grad():
            v = -flow(Mt[va].to(device)).log_prob(Xt[va].to(device)).mean().item()
        hist.append(v)
        if v < best:
            best, best_state = v, {k: t.detach().clone() for k, t in flow.state_dict().items()}
    flow.load_state_dict(best_state)
    return flow, sx, sm, hist


def sample_flow(flow, sx, sm, m, device="cpu"):
    with torch.no_grad():
        c = torch.tensor(sm.fwd(m[:, None]), dtype=torch.float32).to(device)
        z = flow(c).sample().cpu().numpy()
    return sx.inv(z)


def cathode(data, splits, X_test, seed, oversample=4, epochs=60, device="cpu", return_extra=False):
    lo, hi = SR
    sb = data[~in_sr(data.mjj) & (data.mjj > 2.6) & (data.mjj < 4.6)]  # wide sidebands for the density estimate
    flow, sx, sm, hist = train_flow(sb[FEATURES].values, sb.mjj.values, epochs=epochs, seed=seed, device=device)
    d_sr = sr(data)
    # mjj values for the samples: a KDE-free choice, resample the observed SR mjj values.
    rng = np.random.default_rng(seed)
    m_samp = rng.choice(d_sr.mjj.values, size=oversample * len(d_sr))
    synth = sample_flow(flow, sx, sm, m_samp, device)
    ok = np.isfinite(synth).all(1)
    synth = synth[ok]
    scores = classifier_scores(d_sr[FEATURES].values, synth, X_test, seed=seed)
    if return_extra:
        return scores, {"flow_val_nll": hist, "synth": synth}
    return scores


def sic_curve(scores, labels, n_points=200):
    """Significance improvement eps_S / sqrt(eps_B) as a function of signal efficiency."""
    order = np.argsort(-scores)
    l = labels[order]
    tp = np.cumsum(l) / l.sum(); fp = np.cumsum(1 - l) / (1 - l).sum()
    keep = fp > 0
    eff_s, eff_b = tp[keep], fp[keep]
    grid = np.linspace(0.01, 1, n_points)
    idx = np.searchsorted(eff_s, grid)
    idx = np.clip(idx, 0, len(eff_s) - 1)
    sic = eff_s[idx] / np.sqrt(eff_b[idx])
    return grid, sic, eff_s, eff_b


def max_sic(scores, labels, min_bkg_events=20):
    """Max SIC, ignoring the noisy tail where fewer than min_bkg_events test background events pass."""
    order = np.argsort(-scores)
    l = labels[order]
    nb = np.cumsum(1 - l)
    tp = np.cumsum(l) / l.sum(); fp = nb / (1 - l).sum()
    ok = nb >= min_bkg_events
    return float(np.max(tp[ok] / np.sqrt(fp[ok])))
