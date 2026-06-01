import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from captum.attr import Saliency, IntegratedGradients, DeepLift, InputXGradient, LRP
from captum.attr._utils.lrp_rules import EpsilonRule

# -----------------------------
# SETTINGS
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "../result/EEGNet_NoICA_FIXED_FOR_XAI/model.pth"
TEST_DATA_PATH = "../result/EEGNet_NoICA_FIXED_FOR_XAI/xai_fixed_test_data.npz"
PRED_PATH = "../result/EEGNet_NoICA_FIXED_FOR_XAI/xai_fixed_predictions.npz"

RESULT_DIR = "../result/XAI_EEGNet_NoICA_LEFT_RIGHT_FIXED_correct_only"
os.makedirs(RESULT_DIR, exist_ok=True)

NUM_SAMPLES_PER_CLASS = 100

CHANNEL_NAMES = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz"
]

LABEL_NAMES = {
    0: "Left_Hand",
    1: "Right_Hand"
}

# -----------------------------
# MODEL
# -----------------------------
class EEGNet(nn.Module):
    def __init__(self, num_classes=2, chans=22, samples=1001):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=(chans, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(1, 16), padding=(0, 8), groups=16, bias=False),
            nn.Conv2d(16, 16, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, chans, samples)
            out = self.block2(self.block1(dummy))
            self.flatten_dim = out.reshape(1, -1).shape[1]

        self.classifier = nn.Linear(self.flatten_dim, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.reshape(x.size(0), -1)
        return self.classifier(x)

# -----------------------------
# LOAD FIXED TEST DATA + PREDICTIONS
# -----------------------------
test_data = np.load(TEST_DATA_PATH)
pred_data = np.load(PRED_PATH)

X_test = test_data["X_test"]
y_test = pred_data["y_test"]
preds = pred_data["preds"]

print("X_test:", X_test.shape)
print("y_test counts:", np.bincount(y_test.astype(int)))
print("pred counts:", np.bincount(preds.astype(int)))

# Add EEGNet input dimension
X_test = X_test[:, np.newaxis, :, :]
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = EEGNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("Model loaded")

# -----------------------------
# SELECT CORRECT LEFT AND RIGHT SAMPLES
# -----------------------------
correct_mask = preds == y_test

left_correct_idx = np.where((y_test == 0) & correct_mask)[0]
right_correct_idx = np.where((y_test == 1) & correct_mask)[0]

left_selected = left_correct_idx[:NUM_SAMPLES_PER_CLASS]
right_selected = right_correct_idx[:NUM_SAMPLES_PER_CLASS]

print("Correct left-hand available:", len(left_correct_idx))
print("Correct right-hand available:", len(right_correct_idx))
print("Selected left-hand:", len(left_selected))
print("Selected right-hand:", len(right_selected))

# -----------------------------
# LRP RULES
# -----------------------------
for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ELU, nn.AvgPool2d)):
        m.rule = EpsilonRule(epsilon=1e-6)

# -----------------------------
# XAI METHODS
# -----------------------------
methods = {
    "Saliency": Saliency(model),
    "IG": IntegratedGradients(model),
    "DeepLift": DeepLift(model),
    "IxG": InputXGradient(model),
    "LRP": None
}

# -----------------------------
# RUN XAI FOR ONE CLASS
# -----------------------------
def run_xai_for_class(class_label, selected_indices, writer):
    class_name = LABEL_NAMES[class_label]

    print("\n==============================")
    print("Running XAI for:", class_name)
    print("==============================")

    X_class = X_test_tensor[selected_indices]
    target_class = torch.tensor(preds[selected_indices], dtype=torch.long).to(DEVICE)

    class_dir = os.path.join(RESULT_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)

    summary_rows = []

    for method_name, method in methods.items():
        print("Running:", method_name)

        if method_name == "IG":
            attr = method.attribute(X_class, target=target_class, n_steps=50)

        elif method_name == "LRP":
            # Re-assign LRP rules every time before running LRP
            for m in model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ELU, nn.AvgPool2d)):
                    m.rule = EpsilonRule(epsilon=1e-6)

            lrp_method = LRP(model)
            attr = lrp_method.attribute(X_class, target=target_class)

        else:
            attr = method.attribute(X_class, target=target_class)

        attr = np.abs(attr.detach().cpu().numpy())

        # attr shape: samples, 1, channels, time
        channel_imp = attr.mean(axis=(1, 3))  # samples x channels

        avg = channel_imp.mean(axis=0)
        std = channel_imp.std(axis=0)

        # Normalize channel importance
        total = avg.sum()
        avg_norm = avg / total
        std_norm = std / total

        df = pd.DataFrame({
            "Channel": CHANNEL_NAMES,
            "AvgImportance": np.round(avg_norm, 6),
            "StdImportance": np.round(std_norm, 6)
        })

        df = df.sort_values(by="AvgImportance", ascending=False)

        # Save sheet
        sheet_name = f"{class_name}_{method_name}"
        df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

        # Save summary
        summary_rows.append({
            "Class": class_name,
            "Method": method_name,
            "TopChannel": df.iloc[0]["Channel"],
            "TopChannelImportance": df.iloc[0]["AvgImportance"],
            "SecondChannel": df.iloc[1]["Channel"],
            "SecondChannelImportance": df.iloc[1]["AvgImportance"],
            "ThirdChannel": df.iloc[2]["Channel"],
            "ThirdChannelImportance": df.iloc[2]["AvgImportance"],
            "SamplesUsed": len(selected_indices)
        })

        # Plot
        plt.figure(figsize=(12, 4))
        plt.bar(df["Channel"], df["AvgImportance"])
        plt.xticks(rotation=45)
        plt.ylabel("Average normalized importance")
        plt.title(f"{class_name} - {method_name} Channel Importance")
        plt.tight_layout()

        plot_path = os.path.join(class_dir, f"{class_name}_{method_name}_channel_importance.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

    return summary_rows

# -----------------------------
# SAVE EXCEL
# -----------------------------
excel_path = os.path.join(RESULT_DIR, "xai_left_right_fixed_noica_correct_only.xlsx")

all_summary_rows = []

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:

    info_df = pd.DataFrame({
        "Variable": [
            "ModelPath",
            "TestDataPath",
            "PredictionPath",
            "Correct samples only",
            "Left label",
            "Right label",
            "Samples per class requested",
            "Left samples used",
            "Right samples used",
            "XAI methods"
        ],
        "Value": [
            MODEL_PATH,
            TEST_DATA_PATH,
            PRED_PATH,
            "Yes",
            "0 = Left hand",
            "1 = Right hand",
            NUM_SAMPLES_PER_CLASS,
            len(left_selected),
            len(right_selected),
            ", ".join(methods.keys())
        ]
    })

    info_df.to_excel(writer, sheet_name="info", index=False)

    left_summary = run_xai_for_class(0, left_selected, writer)
    right_summary = run_xai_for_class(1, right_selected, writer)

    all_summary_rows.extend(left_summary)
    all_summary_rows.extend(right_summary)

    summary_df = pd.DataFrame(all_summary_rows)
    summary_df.to_excel(writer, sheet_name="summary", index=False)

print("\nDONE")
print("Results saved to:", RESULT_DIR)
print("Excel saved to:", excel_path)