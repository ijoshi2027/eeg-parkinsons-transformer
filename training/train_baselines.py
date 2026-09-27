# baselines for eeg pd classification
# svm, random forest, logistic regression, mlp, 1d cnn
# 5-fold stratified group cv, within-dataset only

import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from scipy.signal import butter, sosfiltfilt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

data = np.load("preprocessed_data_ica.npz", allow_pickle=True)
X_raw = data["X"]
y = data["y"]
sids = data["subject_ids"]
source = data["source"]

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


class CNN1D(nn.Module):
    def __init__(self, n_channels=29, n_samples=1000, n_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=25, stride=5, padding=10),
            nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(32, 64, kernel_size=15, stride=5, padding=5),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(64, 128, kernel_size=10, stride=5, padding=3),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.features(x).squeeze(-1))


class MLP(nn.Module):
    def __init__(self, input_dim=100, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


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


def subj_vote(sid_arr, preds, labels, use_probs=True):
    sp, st = {}, {}
    for sid, pred, lab in zip(sid_arr, preds, labels):
        sp.setdefault(sid, []).append(pred)
        st[sid] = lab

    yt, yp = [], []
    for sid in sp:
        yt.append(st[sid])
        if use_probs:
            yp.append(np.argmax(np.mean(sp[sid], axis=0)))
        else:
            vals, cnts = np.unique(sp[sid], return_counts=True)
            yp.append(vals[np.argmax(cnts)])
    return np.array(yt), np.array(yp)


def pca_reduce(Xtr, Xte):
    Xtr = Xtr.reshape(Xtr.shape[0], -1)
    Xte = Xte.reshape(Xte.shape[0], -1)
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)
    pca = PCA(n_components=100, random_state=42)
    return pca.fit_transform(Xtr), pca.transform(Xte)


def run_sklearn(model_fn, X, y_data, sid_data, ds_name):
    uniq = np.unique(sid_data)
    sy = np.array([y_data[sid_data == s][0] for s in uniq])
    ns = min(5, min(np.bincount(sy)))
    if ns < 2:
        return None

    cv = StratifiedGroupKFold(n_splits=ns, shuffle=True, random_state=42)
    accs = []

    for _, (tr_idx, te_idx) in enumerate(cv.split(uniq, sy, groups=uniq)):
        tr_set = set(uniq[tr_idx])
        te_set = set(uniq[te_idx])
        tr_mask = np.array([s in tr_set for s in sid_data])
        te_mask = np.array([s in te_set for s in sid_data])

        Xtr, Xte = pca_reduce(X[tr_mask], X[te_mask])
        m = model_fn()
        m.fit(Xtr, y_data[tr_mask])
        preds = m.predict(Xte)
        yt, yp = subj_vote(sid_data[te_mask], preds, y_data[te_mask], use_probs=False)
        accs.append(balanced_accuracy_score(yt, yp))

    mean_acc = np.mean(accs)
    print(f"    {ds_name}: {mean_acc:.4f} +/- {np.std(accs):.4f}")
    return mean_acc


def train_nn(model, tr_loader, te_loader, te_sids, te_y):
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)

    best = 0
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

        yt, yp = subj_vote(te_sids, np.array(probs), te_y, use_probs=True)
        acc = balanced_accuracy_score(yt, yp)
        if acc > best:
            best = acc
            wait = 0
        else:
            wait += 1
        if wait >= 15:
            break

    return best


def run_nn(model_fn, X, y_data, sid_data, ds_name, use_pca=False):
    uniq = np.unique(sid_data)
    sy = np.array([y_data[sid_data == s][0] for s in uniq])
    ns = min(5, min(np.bincount(sy)))
    if ns < 2:
        return None

    cv = StratifiedGroupKFold(n_splits=ns, shuffle=True, random_state=42)
    accs = []

    for _, (tr_idx, te_idx) in enumerate(cv.split(uniq, sy, groups=uniq)):
        tr_set = set(uniq[tr_idx])
        te_set = set(uniq[te_idx])
        tr_mask = np.array([s in tr_set for s in sid_data])
        te_mask = np.array([s in te_set for s in sid_data])

        Xtr, Xte = X[tr_mask].copy(), X[te_mask].copy()

        if use_pca:
            Xtr, Xte = pca_reduce(Xtr, Xte)
        else:
            mu = Xtr.mean(axis=0, keepdims=True)
            sigma = Xtr.std(axis=0, keepdims=True) + 1e-8
            Xtr = (Xtr - mu) / sigma
            Xte = (Xte - mu) / sigma

        tr_loader = DataLoader(
            EEGData(Xtr, y_data[tr_mask]), batch_size=64,
            sampler=get_sampler(y_data[tr_mask])
        )
        te_loader = DataLoader(EEGData(Xte, y_data[te_mask]), batch_size=64)

        model = model_fn().to(dev)
        acc = train_nn(model, tr_loader, te_loader, sid_data[te_mask], y_data[te_mask])
        accs.append(acc)

    mean_acc = np.mean(accs)
    print(f"    {ds_name}: {mean_acc:.4f} +/- {np.std(accs):.4f}")
    return mean_acc


iowa_mask = source == "iowa"
sd_mask = source == "sd"

sklearn_models = {
    "svm": lambda: SVC(kernel="rbf", class_weight="balanced", C=1.0, random_state=42),
    "random forest": lambda: RandomForestClassifier(
        n_estimators=500, max_depth=20, class_weight="balanced",
        random_state=42, n_jobs=-1,
    ),
    "logistic reg": lambda: LogisticRegression(
        class_weight="balanced", max_iter=1000, C=1.0, random_state=42,
    ),
}

results = {}

for bname, blist in band_configs.items():
    Xb = X_raw if blist is None else sum(filt[b] for b in blist)

    print(f"\n--- {bname} ---")

    Xi, Xs = Xb[iowa_mask], Xb[sd_mask]
    yi, ys = y[iowa_mask], y[sd_mask]
    si, ss = sids[iowa_mask], sids[sd_mask]

    for mname, mfn in sklearn_models.items():
        print(f"\n  {mname}")
        acc = run_sklearn(mfn, Xi, yi, si, "iowa")
        if acc: results[(mname, bname, "iowa")] = acc
        acc = run_sklearn(mfn, Xs, ys, ss, "san diego")
        if acc: results[(mname, bname, "san diego")] = acc

    print(f"\n  1d cnn")
    cnn_fn = lambda: CNN1D(n_channels=n_ch, n_samples=n_samp)
    acc = run_nn(cnn_fn, Xi, yi, si, "iowa")
    if acc: results[("1d cnn", bname, "iowa")] = acc
    acc = run_nn(cnn_fn, Xs, ys, ss, "san diego")
    if acc: results[("1d cnn", bname, "san diego")] = acc

    print(f"\n  mlp")
    mlp_fn = lambda: MLP(input_dim=100)
    acc = run_nn(mlp_fn, Xi, yi, si, "iowa", use_pca=True)
    if acc: results[("mlp", bname, "iowa")] = acc
    acc = run_nn(mlp_fn, Xs, ys, ss, "san diego", use_pca=True)
    if acc: results[("mlp", bname, "san diego")] = acc

all_models = ["svm", "random forest", "logistic reg", "1d cnn", "mlp"]

print(f"\n--- within-dataset results (5-fold cv) ---")
for ds in ["iowa", "san diego"]:
    print(f"\n  {ds}:")
    header = f"  {'band':<20}" + "".join(f" {m:>12}" for m in all_models)
    print(header)
    print("  " + "-" * len(header))
    for band in band_configs:
        row = f"  {band:<20}"
        for m in all_models:
            val = results.get((m, band, ds))
            row += f" {val:>12.4f}" if val else f" {'n/a':>12}"
        print(row)
