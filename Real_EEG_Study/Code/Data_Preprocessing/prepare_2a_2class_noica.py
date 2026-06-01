import os
import numpy as np
import mne

# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42
np.random.seed(SEED)

# -----------------------------
# Paths
# -----------------------------
base_dir = ".."
dataset_dir = os.path.join(base_dir, "dataset")
result_dir = os.path.join(base_dir, "result", "NoICA_check")
os.makedirs(result_dir, exist_ok=True)

# -----------------------------
# Subjects: training files only
# -----------------------------
subjects = [
    "A01T", "A02T", "A03T", "A04T", "A05T",
    "A06T", "A07T", "A08T", "A09T"
]

all_X = []
all_y = []

for subject in subjects:
    print("\n===================================")
    print(f"Processing {subject} WITHOUT ICA")
    print("===================================")

    gdf_file = os.path.join(dataset_dir, f"{subject}.gdf")

    raw = mne.io.read_raw_gdf(gdf_file, preload=True, verbose=False)

    raw.set_channel_types({
        "EOG-left": "eog",
        "EOG-central": "eog",
        "EOG-right": "eog"
    })

    # Use only 22 EEG channels, but do NOT apply ICA
    raw_eeg = raw.copy().pick_types(eeg=True, eog=False)

    events, event_id = mne.events_from_annotations(raw)

    selected_event_id = {
        "left_hand": event_id["769"],
        "right_hand": event_id["770"]
    }

    selected_events = events[
        np.isin(events[:, 2], list(selected_event_id.values()))
    ]

    epochs_noica = mne.Epochs(
        raw_eeg,
        selected_events,
        event_id=selected_event_id,
        tmin=0,
        tmax=4,
        picks="eeg",
        baseline=None,
        preload=True,
        verbose=False
    )

    X = epochs_noica.get_data()

    y = np.array([
        0 if event_code == selected_event_id["left_hand"] else 1
        for event_code in epochs_noica.events[:, 2]
    ])

    print("Subject X shape:", X.shape)
    print("Subject y shape:", y.shape)
    print("Subject class counts:", np.bincount(y))

    all_X.append(X)
    all_y.append(y)

# -----------------------------
# Combine all subjects
# -----------------------------
X_all = np.concatenate(all_X, axis=0)
y_all = np.concatenate(all_y, axis=0)

print("\n===================================")
print("FINAL NOICA DATASET")
print("===================================")
print("X_all shape:", X_all.shape)
print("y_all shape:", y_all.shape)
print("Unique labels:", np.unique(y_all))
print("Class counts:", np.bincount(y_all))

# -----------------------------
# Save final NoICA dataset
# -----------------------------
save_path = os.path.join(dataset_dir, "bci2a_2class_NoICA_all_subjects.npz")

np.savez(
    save_path,
    X=X_all,
    y=y_all,
    ch_names=raw_eeg.ch_names,
    sfreq=raw_eeg.info["sfreq"]
)

print("\nSaved final NoICA dataset:")
print(save_path)