# standalone transformer for eeg pd classification
# each channel = one token, linear projection to 64-dim embedding,
# 2-layer transformer encoder w/ 4 heads
# 5-fold stratified group cv on combined dataset

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

n_ep, n_ch, n_samp = X_raw.shape
sfreq = 250.0
print(f"data: {X_raw.shape}, PD: {(y == 1).sum()}, HC: {(y == 0).sum()}")

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


class TransformerClassifier(nn.Module):
    def __init__(self, n_samples, d_model=64, n_heads=4, n_layers=2,
                 dropout=0.3, n_classes=2):
        super().__init__()
        self.proj = nn.Linear(n_samples, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(0.5), nn.Linear(d_model, n_classes),
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.head(x)


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


# cv splits
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
uniq_subj = np.unique(sids)
subj_y = np.array([y[sids == s][0] for s in uniq_subj])
folds = list(cv.split(uniq_subj, subj_y, groups=uniq_subj))

os.makedirs("models", exist_ok=True)


def run(band_name, X):
    print(f"\n--- {band_name} ---")

    fold_acc = []
    fold_states = {}

    for fi, (tr_idx, te_idx) in enumerate(folds):
        tr_set = set(uniq_subj[tr_idx])
        te_set = set(uniq_subj[te_idx])
        tr_mask = np.array([s in tr_set for s in sids])
        te_mask = np.array([s in te_set for s in sids])

        Xtr, Xte = X[tr_mask].copy(), X[te_mask].copy()
        ytr, yte = y[tr_mask], y[te_mask]
        sids_te = sids[te_mask]

        mu = Xtr.mean(axis=0, keepdims=True)
        sigma = Xtr.std(axis=0, keepdims=True) + 1e-8
        Xtr = (Xtr - mu) / sigma
        Xte = (Xte - mu) / sigma

        tr_loader = DataLoader(EEGData(Xtr, ytr), batch_size=64, sampler=get_sampler(ytr))
        te_loader = DataLoader(EEGData(Xte, yte), batch_size=64)

        model = TransformerClassifier(n_samples=n_samp).to(dev)
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

            probs = np.array(probs)
            sp, st = {}, {}
            for sid, p, lab in zip(sids_te, probs, yte):
                sp.setdefault(sid, []).append(p)
                st[sid] = lab
            yt = [st[s] for s in sp]
            yp = [np.argmax(np.mean(sp[s], axis=0)) for s in sp]
            acc = balanced_accuracy_score(yt, yp)

            if acc > best:
                best = acc
                best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
            if wait >= 15:
                break

        print(f"  fold {fi+1}: {best:.4f} (ep {ep+1})")
        fold_acc.append(best)
        fold_states[fi] = best_sd

    mean_acc = np.mean(fold_acc)
    print(f"  {band_name}: {mean_acc:.4f} +/- {np.std(fold_acc):.4f}")

    # save best fold
    best_fi = np.argmax(fold_acc)
    tag = band_name.replace("+", "_").replace(" ", "_")
    path = f"models/{tag}_{mean_acc:.4f}_transformer.pt"
    torch.save({
        "model_state_dict": fold_states[best_fi],
        "band": band_name,
        "fold": best_fi + 1,
        "bal_acc": fold_acc[best_fi],
        "mean_bal_acc": mean_acc,
        "n_samples": n_samp,
    }, path)
    print(f"  saved: {path}")
    return mean_acc


results = {}

for name in bands:
    results[name] = run(name, filt[name])

results["all-band"] = run("all-band", X_raw)

combos = {
    "delta+theta": ["delta", "theta"],
    "theta+alpha": ["theta", "alpha"],
    "theta+beta": ["theta", "beta"],
    "delta+theta+alpha": ["delta", "theta", "alpha"],
    "delta+theta+alpha+beta": ["delta", "theta", "alpha", "beta"],
}
for cname, blist in combos.items():
    results[cname] = run(cname, sum(filt[b] for b in blist))

print(f"\n--- results (transformer, combined dataset) ---")
for name, acc in results.items():
    print(f"  {name:<30} {acc:.4f}")
