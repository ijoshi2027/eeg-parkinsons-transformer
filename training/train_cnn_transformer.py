# cnn-transformer for eeg pd classification
# 3-stage 1d cnn compresses temporal dim, then 2-layer transformer
# w/ classification token does the final prediction
# within-dataset 5-fold cv + cross-dataset transfer

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from scipy.signal import butter, sosfiltfilt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score

data = np.load("preprocessed_data_ica.npz", allow_pickle=True)
X_raw = data["X"]
y = data["y"]
sids = data["subject_ids"]
source = data["source"]

n_ep, n_ch, n_samp = X_raw.shape
sfreq = 250.0

print(f"data: {X_raw.shape}, PD: {(y == 1).sum()}, HC: {(y == 0).sum()}")
for s in np.unique(source):
    mask = source == s
    print(f"  {s}: {mask.sum()} epochs, {len(np.unique(sids[mask]))} subjects")

bands = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}


def bp_filter(X, fmin, fmax):
    sos = butter(4, [fmin, fmax], btype="bandpass", fs=sfreq, output="sos")
    out = np.zeros_like(X)
    for i in range(X.shape[0]):
        for ch in range(X.shape[1]):
            out[i, ch, :] = sosfiltfilt(sos, X[i, ch, :])
    return out


print("filtering bands...")
filt = {}
for name, (lo, hi) in bands.items():
    print(f"  {name} ({lo}-{hi} Hz)")
    filt[name] = bp_filter(X_raw, lo, hi)
print("done.\n")

if torch.cuda.is_available():
    dev = torch.device("cuda")
elif torch.backends.mps.is_available():
    dev = torch.device("mps")
else:
    dev = torch.device("cpu")
print(f"device: {dev}")


class CNNTransformer(nn.Module):
    def __init__(self, n_channels=29, n_samples=1000, d_model=64,
                 n_heads=4, n_layers=2, dropout=0.3, n_classes=2):
        super().__init__()

        self.cnn = nn.Sequential(
            # depthwise: (B, 29, 1000) -> (B, 116, 200)
            nn.Conv1d(n_channels, n_channels * 4, kernel_size=25, stride=5,
                      padding=10, groups=n_channels),
            nn.BatchNorm1d(n_channels * 4), nn.GELU(), nn.Dropout(dropout),
            # temporal: (B, 116, 200) -> (B, 64, 40)
            nn.Conv1d(n_channels * 4, d_model, kernel_size=15, stride=5, padding=5),
            nn.BatchNorm1d(d_model), nn.GELU(), nn.Dropout(dropout),
            # compression: (B, 64, 40) -> (B, 64, 8)
            nn.Conv1d(d_model, d_model, kernel_size=10, stride=5, padding=3),
            nn.BatchNorm1d(d_model), nn.GELU(),
        )

        self.cls_tok = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        T = self._calc_tokens(n_samples)
        self.pos = nn.Parameter(torch.randn(1, T + 1, d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(0.5), nn.Linear(d_model, n_classes),
        )

    def _calc_tokens(self, n):
        L = (n + 2 * 10 - 25) // 5 + 1
        L = (L + 2 * 5 - 15) // 5 + 1
        L = (L + 2 * 3 - 10) // 5 + 1
        return L

    def forward(self, x):
        B = x.shape[0]
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        cls = self.cls_tok.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos
        x = self.encoder(x)
        return self.head(x[:, 0, :])


class EEGData(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.X[i], self.y[i]


def get_sampler(labels):
    counts = np.bincount(labels)
    w = np.array([1.0 / counts[l] for l in labels])
    return WeightedRandomSampler(w, num_samples=len(labels), replacement=True)


def subj_vote(sids, probs, labels):
    sp, st = {}, {}
    for sid, p, lab in zip(sids, probs, labels):
        sp.setdefault(sid, []).append(p)
        st[sid] = lab
    yt = [st[s] for s in sp]
    yp = [np.argmax(np.mean(sp[s], axis=0)) for s in sp]
    return yt, yp


def train_eval(model, tr_loader, te_loader, te_sids, te_y):
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)

    best = 0
    best_sd = None
    wait = 0

    for ep in range(60):
        model.train()
        for bx, by in tr_loader:
            bx, by = bx.to(dev), by.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        probs = []
        with torch.no_grad():
            for bx, _ in te_loader:
                out = model(bx.to(dev))
                probs.extend(F.softmax(out, dim=1).cpu().numpy())

        yt, yp = subj_vote(te_sids, np.array(probs), te_y)
        acc = balanced_accuracy_score(yt, yp)

        if acc > best:
            best = acc
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= 15:
            break

    return best, best_sd


def run_within(ds_name, X, y_data, sid_data, band):
    uniq = np.unique(sid_data)
    sy = np.array([y_data[sid_data == s][0] for s in uniq])
    n_splits = min(5, min(np.bincount(sy)))
    if n_splits < 2:
        print(f"  {ds_name}: skipped (not enough subjects)")
        return None

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []

    for fi, (tr_idx, te_idx) in enumerate(cv.split(uniq, sy, groups=uniq), 1):
        tr_set = set(uniq[tr_idx])
        te_set = set(uniq[te_idx])
        tr_mask = np.array([s in tr_set for s in sid_data])
        te_mask = np.array([s in te_set for s in sid_data])

        Xtr, Xte = X[tr_mask].copy(), X[te_mask].copy()
        ytr, yte = y_data[tr_mask], y_data[te_mask]

        mu = Xtr.mean(axis=0, keepdims=True)
        sigma = Xtr.std(axis=0, keepdims=True) + 1e-8
        Xtr = (Xtr - mu) / sigma
        Xte = (Xte - mu) / sigma

        tr_loader = DataLoader(EEGData(Xtr, ytr), batch_size=64, sampler=get_sampler(ytr))
        te_loader = DataLoader(EEGData(Xte, yte), batch_size=64)

        model = CNNTransformer(n_channels=X.shape[1], n_samples=X.shape[2]).to(dev)
        acc, _ = train_eval(model, tr_loader, te_loader, sid_data[te_mask], yte)
        print(f"  fold {fi}: {acc:.4f}")
        accs.append(acc)

    mean_acc = np.mean(accs)
    print(f"  {ds_name} ({band}): {mean_acc:.4f} +/- {np.std(accs):.4f}")
    return mean_acc


def run_cross(tr_name, te_name, Xtr, ytr, Xte, yte, te_sids, band):
    print(f"\n  cross: {tr_name} -> {te_name} | {band}")

    mu = Xtr.mean(axis=0, keepdims=True)
    sigma = Xtr.std(axis=0, keepdims=True) + 1e-8
    Xtr_n = (Xtr - mu) / sigma
    Xte_n = (Xte - mu) / sigma

    tr_loader = DataLoader(EEGData(Xtr_n, ytr), batch_size=64, sampler=get_sampler(ytr))
    te_loader = DataLoader(EEGData(Xte_n, yte), batch_size=64)

    model = CNNTransformer(n_channels=Xtr.shape[1], n_samples=Xtr.shape[2]).to(dev)
    acc, _ = train_eval(model, tr_loader, te_loader, te_sids, yte)
    print(f"  {tr_name} -> {te_name}: {acc:.4f}")
    return acc


os.makedirs("models", exist_ok=True)

iowa_mask = source == "iowa"
sd_mask = source == "sd"

band_configs = {
    "delta": ["delta"],
    "theta": ["theta"],
    "alpha": ["alpha"],
    "beta": ["beta"],
    "gamma": ["gamma"],
    "delta+theta": ["delta", "theta"],
    "theta+alpha": ["theta", "alpha"],
    "theta+beta": ["theta", "beta"],
    "delta+theta+alpha": ["delta", "theta", "alpha"],
    "all-band": None,
}

within_res = {}
cross_res = {}

for bname, blist in band_configs.items():
    Xb = X_raw if blist is None else sum(filt[b] for b in blist)

    print(f"\n--- {bname} ---")

    acc = run_within("iowa", Xb[iowa_mask], y[iowa_mask], sids[iowa_mask], bname)
    if acc is not None:
        within_res[f"iowa ({bname})"] = acc

    acc = run_within("san diego", Xb[sd_mask], y[sd_mask], sids[sd_mask], bname)
    if acc is not None:
        within_res[f"san diego ({bname})"] = acc

    acc = run_cross("iowa", "san diego",
                    Xb[iowa_mask], y[iowa_mask],
                    Xb[sd_mask], y[sd_mask],
                    sids[sd_mask], bname)
    cross_res[f"iowa->sd ({bname})"] = acc

    acc = run_cross("san diego", "iowa",
                    Xb[sd_mask], y[sd_mask],
                    Xb[iowa_mask], y[iowa_mask],
                    sids[iowa_mask], bname)
    cross_res[f"sd->iowa ({bname})"] = acc

print(f"\n--- within-dataset results ---")
for name, acc in within_res.items():
    print(f"  {name:<35} {acc:.4f}")

print(f"\n--- cross-dataset results ---")
for name, acc in cross_res.items():
    print(f"  {name:<35} {acc:.4f}")
