import os
import numpy as np
import mne
from mne.preprocessing import ICA
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
base_dir = ".."
dataset_dir = os.path.join(base_dir, "dataset")
result_dir = os.path.join(base_dir, "result", "ICA_component_visualization")
os.makedirs(result_dir, exist_ok=True)

# -----------------------------
# Use one subject first
# -----------------------------
subject = "A01T"
gdf_file = os.path.join(dataset_dir, f"{subject}.gdf")

print("Loading:", gdf_file)

raw = mne.io.read_raw_gdf(gdf_file, preload=True, verbose=False)

raw.set_channel_types({
    "EOG-left": "eog",
    "EOG-central": "eog",
    "EOG-right": "eog"
})

# -----------------------------
# EEG only for ICA fitting
# -----------------------------
raw_eeg = raw.copy().pick_types(eeg=True, eog=False)
raw_eeg.filter(l_freq=1.0, h_freq=40.0, verbose=False)

# -----------------------------
# Fit ICA
# -----------------------------
ica = ICA(
    n_components=22,
    random_state=42,
    max_iter="auto"
)

print("Fitting ICA...")
ica.fit(raw_eeg)

# -----------------------------
# Detect EOG-related components
# -----------------------------
eog_indices, eog_scores = ica.find_bads_eog(
    raw,
    ch_name="EOG-central",
    threshold=2.0
)

print("Detected EOG-related components:", eog_indices)

# -----------------------------
# 1. Plot ICA component time series
# -----------------------------
sources = ica.get_sources(raw_eeg)
source_data = sources.get_data()

sfreq = raw_eeg.info["sfreq"]

start_sec = 100
duration_sec = 10

start_sample = int(start_sec * sfreq)
end_sample = int((start_sec + duration_sec) * sfreq)

time_axis = np.arange(end_sample - start_sample) / sfreq

# Plot all ICA components
plt.figure(figsize=(14, 20))

for i in range(22):
    plt.subplot(22, 1, i + 1)
    plt.plot(time_axis, source_data[i, start_sample:end_sample])

    title = f"ICA Component {i}"
    if i in eog_indices:
        title += "  <-- EOG-related component"

    plt.title(title, fontsize=8)
    plt.ylabel("Amp")

plt.xlabel("Time (seconds)")
plt.tight_layout()

all_components_path = os.path.join(result_dir, f"{subject}_all_ICA_components.png")
plt.savefig(all_components_path, dpi=300)
plt.close()

print("Saved all ICA component plot:")
print(all_components_path)

# -----------------------------
# 2. Plot only detected EOG components
# -----------------------------
if len(eog_indices) > 0:
    plt.figure(figsize=(12, 4 * len(eog_indices)))

    for idx, comp in enumerate(eog_indices):
        plt.subplot(len(eog_indices), 1, idx + 1)
        plt.plot(time_axis, source_data[comp, start_sample:end_sample])
        plt.title(f"{subject} - EOG-related ICA Component {comp}")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")

    plt.tight_layout()

    eog_component_path = os.path.join(result_dir, f"{subject}_EOG_related_components.png")
    plt.savefig(eog_component_path, dpi=300)
    plt.close()

    print("Saved EOG-related component plot:")
    print(eog_component_path)

# -----------------------------
# 3. Remove EOG component and reconstruct clean EEG
# -----------------------------
raw_before = raw.copy().pick_types(eeg=True, eog=False)
raw_after = raw_before.copy()

ica.exclude = eog_indices
ica.apply(raw_after)

# -----------------------------
# 4. Plot before vs after reconstructed EEG
# -----------------------------
channel_name = "EEG-Fz"
channel_index = raw_before.ch_names.index(channel_name)

before_signal = raw_before.get_data()[channel_index, start_sample:end_sample]
after_signal = raw_after.get_data()[channel_index, start_sample:end_sample]

plt.figure(figsize=(12, 5))
plt.plot(time_axis, before_signal, label="Before ICA")
plt.plot(time_axis, after_signal, label="After ICA")
plt.title(f"{subject} - EEG-Fz Before and After Removing EOG Component")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()

overlay_path = os.path.join(result_dir, f"{subject}_before_after_EOG_removal.png")
plt.savefig(overlay_path, dpi=300)
plt.close()

print("Saved before/after overlay plot:")
print(overlay_path)

print("\nDone.")