import os
import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
base_dir = ".."
dataset_dir = os.path.join(base_dir, "dataset")
result_dir = os.path.join(base_dir, "result", "Auto_ICA_EOG_Visualization")
os.makedirs(result_dir, exist_ok=True)

subjects = [
    "A01T", "A02T", "A03T", "A04T", "A05T",
    "A06T", "A07T", "A08T", "A09T"
]

summary_rows = []

for subject in subjects:
    print("\n===================================")
    print(f"Processing {subject}")
    print("===================================")

    gdf_file = os.path.join(dataset_dir, f"{subject}.gdf")

    raw = mne.io.read_raw_gdf(gdf_file, preload=True, verbose=False)

    raw.set_channel_types({
        "EOG-left": "eog",
        "EOG-central": "eog",
        "EOG-right": "eog"
    })

    # EEG only for ICA
    raw_eeg = raw.copy().pick_types(eeg=True, eog=False)
    raw_eeg.filter(l_freq=1.0, h_freq=40.0, verbose=False)

    # Fit ICA
    ica = ICA(
        n_components=22,
        random_state=42,
        max_iter="auto"
    )

    print("Fitting ICA...")
    ica.fit(raw_eeg)

    # Automatically detect EOG-related components
    eog_indices, eog_scores = ica.find_bads_eog(
        raw,
        ch_name="EOG-central",
        threshold=2.0
    )

    print("Auto-detected EOG components:", eog_indices)

    # Save summary row
    summary_rows.append({
        "Subject": subject,
        "Auto_EOG_Components": str(eog_indices),
        "Num_Components_Removed": len(eog_indices)
    })

    # Get ICA source signals
    sources = ica.get_sources(raw_eeg)
    source_data = sources.get_data()

    sfreq = raw_eeg.info["sfreq"]
    start_sec = 100
    duration_sec = 10

    start_sample = int(start_sec * sfreq)
    end_sample = int((start_sec + duration_sec) * sfreq)

    time_axis = np.arange(end_sample - start_sample) / sfreq

    # -----------------------------
    # Plot detected EOG ICA sources
    # -----------------------------
    if len(eog_indices) > 0:
        plt.figure(figsize=(14, 4 * len(eog_indices)))

        for i, comp in enumerate(eog_indices):
            plt.subplot(len(eog_indices), 1, i + 1)
            plt.plot(
                time_axis,
                source_data[comp, start_sample:end_sample],
                color="blue"
            )
            plt.title(f"{subject} - Auto-detected EOG ICA Source / Component {comp}")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Source amplitude")

        plt.tight_layout()

        source_plot_path = os.path.join(
            result_dir,
            f"{subject}_auto_EOG_sources.png"
        )
        plt.savefig(source_plot_path, dpi=300)
        plt.close()

        print("Saved source plot:", source_plot_path)

    else:
        print("No EOG component detected for", subject)

    # -----------------------------
    # Apply ICA removal
    # -----------------------------
    raw_before = raw.copy().pick_types(eeg=True, eog=False)
    raw_after = raw_before.copy()

    ica.exclude = eog_indices
    ica.apply(raw_after)

    # -----------------------------
    # Plot BEFORE and AFTER on EEG-Fz
    # -----------------------------
    channel_name = "EEG-Fz"

    if channel_name not in raw_before.ch_names:
        channel_name = raw_before.ch_names[0]

    channel_index = raw_before.ch_names.index(channel_name)

    before_signal = raw_before.get_data()[channel_index, start_sample:end_sample]
    after_signal = raw_after.get_data()[channel_index, start_sample:end_sample]

    # Individual stacked before/after plot
    plt.figure(figsize=(14, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time_axis, before_signal, color="blue")
    plt.title(f"{subject} - BEFORE ICA - {channel_name}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.subplot(2, 1, 2)
    plt.plot(time_axis, after_signal, color="orange")
    plt.title(f"{subject} - AFTER ICA - Auto Removed {eog_indices}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.tight_layout()

    before_after_path = os.path.join(
        result_dir,
        f"{subject}_auto_before_after_EOG_removed.png"
    )

    plt.savefig(before_after_path, dpi=300)
    plt.close()

    print("Saved before/after plot:", before_after_path)

# -----------------------------
# Save Excel summary
# -----------------------------
summary_df = pd.DataFrame(summary_rows)

summary_path = os.path.join(result_dir, "auto_eog_component_summary.xlsx")
summary_df.to_excel(summary_path, index=False)

print("\nSaved auto ICA EOG summary:")
print(summary_path)

print("\nDone.")