# preprocessing for iowa + san diego eeg datasets
# downloads (run in terminal):
#   pip install mne mne-icalabel pandas awscli
#   aws s3 sync --no-sign-request s3://openneuro.org/ds004584 ./UOfIowaDataset/
#   aws s3 sync --no-sign-request s3://openneuro.org/ds002778 ./UCSanDiegoDataset/

import mne
import numpy as np
import pandas as pd
from mne_icalabel import label_components

sfreq = 250.0
epoch_dur = 4.0
montage = mne.channels.make_standard_montage("standard_1005")


def preprocess(raw, ch_list):
    raw.pick_channels(ch_list)
    raw.reorder_channels(ch_list)
    raw.crop(tmin=0, tmax=120.0)
    raw.resample(sfreq)
    raw.filter(1.0, 45.0)
    raw.set_montage(montage, match_case=False, on_missing="warn")
    raw.set_eeg_reference("average")

    # ica artifact rejection w/ iclabel
    ica = mne.preprocessing.ICA(n_components=15, random_state=42)
    ica.fit(raw)

    labels = label_components(raw, ica, method="iclabel")
    comp_labels = labels["labels"]
    comp_probs = labels["y_pred_proba"]

    # toss non-brain components w/ >90% confidence, max 5
    exclude = []
    for idx, (label, probs) in enumerate(zip(comp_labels, comp_probs)):
        if label != "brain" and np.max(probs) > 0.9:
            exclude.append(idx)
    exclude = exclude[:5]

    ica.exclude = exclude
    print(f"  removing {len(exclude)} ICA components")
    ica.apply(raw)

    # per-channel z-score
    data = raw.get_data()
    means = data.mean(axis=1, keepdims=True)
    stds = data.std(axis=1, keepdims=True)
    raw._data = (data - means) / (stds + 1e-12)

    return raw


# find 29 channels common to both datasets
iowa_raw = mne.io.read_raw_eeglab(
    "UOfIowaDataset/sub-001/eeg/sub-001_task-Rest_eeg.set", preload=True
)
sd_raw = mne.io.read_raw_bdf(
    "UCSanDiegoDataset/sub-hc1/ses-hc/eeg/sub-hc1_ses-hc_task-rest_eeg.bdf",
    preload=True,
)

iowa_common_ch = set(iowa_raw.ch_names)
for pID in range(1, 150):
    padded = str(pID).zfill(3)
    path = f"UOfIowaDataset/sub-{padded}/eeg/sub-{padded}_task-Rest_eeg.set"
    raw = mne.io.read_raw_eeglab(path, preload=True)
    iowa_common_ch &= set(raw.ch_names)

common_ch = sorted(iowa_common_ch & set(sd_raw.ch_names))
print(f"common channels ({len(common_ch)}): {common_ch}")

all_data = []
all_labels = []
all_sids = []
all_source = []

# iowa (subjects 1-100: PD, 101-149: HC)
for pID in range(1, 150):
    padded = str(pID).zfill(3)
    print(f"iowa subject {padded}")
    path = f"UOfIowaDataset/sub-{padded}/eeg/sub-{padded}_task-Rest_eeg.set"
    raw = mne.io.read_raw_eeglab(path, preload=True)
    preprocessed = preprocess(raw, common_ch)
    epochs = mne.make_fixed_length_epochs(preprocessed, duration=epoch_dur, preload=True)

    all_data.append(epochs.get_data().astype(np.float32))
    label = 1 if pID <= 100 else 0
    all_labels.append([label] * len(epochs))
    all_sids.append([f"sub-{padded}_iowa"] * len(epochs))
    all_source.append(["iowa"] * len(epochs))

# san diego (subject ID contains "pd" or "hc")
df = pd.read_csv("UCSanDiegoDataset/participants.tsv", sep="\t")
for pID in df.iloc[:, 0].tolist():
    print(f"san diego subject {pID}")
    if "pd" in pID:
        path = f"UCSanDiegoDataset/{pID}/ses-off/eeg/{pID}_ses-off_task-rest_eeg.bdf"
    else:
        path = f"UCSanDiegoDataset/{pID}/ses-hc/eeg/{pID}_ses-hc_task-rest_eeg.bdf"
    raw = mne.io.read_raw_bdf(path, preload=True)
    preprocessed = preprocess(raw, common_ch)
    epochs = mne.make_fixed_length_epochs(preprocessed, duration=epoch_dur, preload=True)

    all_data.append(epochs.get_data().astype(np.float32))
    label = 1 if "pd" in pID else 0
    all_labels.append([label] * len(epochs))
    all_sids.append([pID] * len(epochs))
    all_source.append(["sd"] * len(epochs))

X = np.concatenate(all_data, axis=0)
y = np.concatenate(all_labels, axis=0)
sids = np.concatenate(all_sids, axis=0)
src = np.concatenate(all_source, axis=0)

print(f"\nfinal shape: {X.shape}, PD: {(y == 1).sum()}, HC: {(y == 0).sum()}")

np.savez("preprocessed_data_ica.npz", X=X, y=y, subject_ids=sids, source=src)
print("saved preprocessed_data_ica.npz")
