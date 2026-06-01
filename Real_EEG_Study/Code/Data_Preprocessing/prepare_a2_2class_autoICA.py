import os
import mne
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = "../dataset"
SAVE_PATH = "../dataset/bci2a_2class_autoICA_all_subjects.npz"

SUBJECTS = [f"A0{i}T.gdf" for i in range(1, 10)]

X_all = []
y_all = []

# -----------------------------
# LOOP SUBJECTS
# -----------------------------
for subj in SUBJECTS:
    print(f"\nProcessing {subj}...")

    file_path = os.path.join(DATA_PATH, subj)

    # Load raw GDF
    raw = mne.io.read_raw_gdf(file_path, preload=True)

    # Set EOG channel types
    raw.set_channel_types({
        "EOG-left": "eog",
        "EOG-central": "eog",
        "EOG-right": "eog"
    })

    # EEG only for ICA and final dataset
    raw_eeg = raw.copy().pick_types(eeg=True, eog=False)

    # Filter EEG before ICA
    raw_eeg.filter(1., 40., fir_design="firwin")

    # -----------------------------
    # ICA
    # -----------------------------
    ica = mne.preprocessing.ICA(
        n_components=15,
        random_state=42,
        max_iter="auto"
    )

    print("Fitting ICA...")
    ica.fit(raw_eeg)

    # Auto-detect EOG components using EOG-central
    eog_inds, scores = ica.find_bads_eog(
        raw,
        ch_name="EOG-central"
    )

    print(f"Auto detected EOG components: {eog_inds}")

    # Remove EOG components from EEG data
    raw_clean = raw_eeg.copy()
    ica.exclude = eog_inds
    ica.apply(raw_clean)

    # -----------------------------
    # EVENTS
    # -----------------------------
    events, event_dict = mne.events_from_annotations(raw)

    print("Event dictionary:", event_dict)

    event_id_2class = {
        "769": event_dict["769"],   # left hand
        "770": event_dict["770"]    # right hand
    }

    epochs = mne.Epochs(
        raw_clean,
        events,
        event_id=event_id_2class,
        tmin=0,
        tmax=4,
        baseline=None,
        preload=True
    )

    X = epochs.get_data()
    y = epochs.events[:, -1]

    # Convert labels:
    # 769 / left hand  -> 0
    # 770 / right hand -> 1
    y = np.where(y == event_dict["769"], 0, 1)

    print("Subject X shape:", X.shape)
    print("Subject y shape:", y.shape)
    print("Class counts:", np.bincount(y))

    X_all.append(X)
    y_all.append(y)

# -----------------------------
# MERGE ALL SUBJECTS
# -----------------------------
X_all = np.concatenate(X_all, axis=0)
y_all = np.concatenate(y_all, axis=0)

print("\nFinal shape:")
print("X:", X_all.shape)
print("y:", y_all.shape)
print("Final class counts:", np.bincount(y_all))

# -----------------------------
# SAVE
# -----------------------------
np.savez(
    SAVE_PATH,
    X=X_all,
    y=y_all
)

print("\nSaved successfully:")
print(SAVE_PATH)