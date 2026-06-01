import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# -----------------------------
# Seeds to run
# -----------------------------
SEEDS = [87, 75, 43]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Paths
# -----------------------------
base_dir = ".."
dataset_dir = os.path.join(base_dir, "dataset")
result_dir = os.path.join(base_dir, "result")

ica_file = os.path.join(dataset_dir, "bci2a_2class_autoICA_all_subjects.npz")
noica_file = os.path.join(dataset_dir, "bci2a_2class_NoICA_all_subjects.npz")

# -----------------------------
# RNN-LSTM Model
# Input shape: (N, time, channels)
# -----------------------------
class EEG_LSTM(nn.Module):
    def __init__(self, input_size=22, hidden_size=64, num_layers=2, num_classes=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        return self.classifier(last_out)

# -----------------------------
# One fold training
# -----------------------------
def train_one_fold(X_train_full, y_train_full, X_test, y_test, fold_num, seed, epochs=30):

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=seed,
        stratify=y_train_full
    )

    print("Train shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Test shape:", X_test.shape)

    # Normalize using training data only
    mean = X_train.mean()
    std = X_train.std()

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    # Reshape for LSTM: (N, channels, time) -> (N, time, channels)
    X_train = np.transpose(X_train, (0, 2, 1))
    X_val = np.transpose(X_val, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))

    print("LSTM X_train shape:", X_train.shape)
    print("LSTM X_val shape:", X_val.shape)
    print("LSTM X_test shape:", X_test.shape)

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

    set_seed(seed)
    model = EEG_LSTM().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    final_train_loss = 0
    final_train_acc = 0

    for epoch in range(epochs):
        model.train()

        running_loss = 0
        correct = 0
        total = 0

        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            outputs = model(Xb)
            loss = criterion(outputs, yb)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, pred = torch.max(outputs, 1)
            total += yb.size(0)
            correct += (pred == yb).sum().item()

        final_train_loss = running_loss / len(train_loader)
        final_train_acc = correct / total

        print(
            f"Seed {seed} | Fold {fold_num} | Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {final_train_loss:.4f} | Train Acc: {final_train_acc:.4f}"
        )

    # Validation
    model.eval()
    val_loss_total = 0
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            outputs = model(Xb)
            loss = criterion(outputs, yb)
            val_loss_total += loss.item()

            _, pred = torch.max(outputs, 1)
            val_preds.extend(pred.cpu().numpy())
            val_labels.extend(yb.cpu().numpy())

    val_loss = val_loss_total / len(val_loader)
    val_acc = accuracy_score(val_labels, val_preds)

    # Test
    test_loss_total = 0
    test_preds = []
    test_labels = []

    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            outputs = model(Xb)
            loss = criterion(outputs, yb)
            test_loss_total += loss.item()

            _, pred = torch.max(outputs, 1)
            test_preds.extend(pred.cpu().numpy())
            test_labels.extend(yb.cpu().numpy())

    test_loss = test_loss_total / len(test_loader)
    test_acc = accuracy_score(test_labels, test_preds)
    f1 = f1_score(test_labels, test_preds, average="weighted", zero_division=0)
    precision = precision_score(test_labels, test_preds, average="weighted", zero_division=0)
    recall = recall_score(test_labels, test_preds, average="weighted", zero_division=0)

    return {
        "fold": f"Fold-{fold_num}",
        "train_loss": final_train_loss,
        "train_acc": final_train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "f1-score": f1,
        "precision": precision,
        "recall": recall
    }

# -----------------------------
# Run 5-fold for one seed/config
# -----------------------------
def run_5fold(data_path, save_folder, config_name, seed):

    set_seed(seed)
    os.makedirs(save_folder, exist_ok=True)

    data = np.load(data_path)
    X = data["X"]
    y = data["y"]

    print("\n===================================")
    print(f"Running RNN-LSTM 5-Fold: {config_name}")
    print(f"Seed: {seed}")
    print("Dataset:", data_path)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("===================================")

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed
    )

    fold_results = []

    for fold_num, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):

        print("\n-----------------------------------")
        print(f"Starting Seed {seed} | {config_name} | Fold {fold_num}")
        print("-----------------------------------")

        X_train_full = X[train_idx]
        X_test = X[test_idx]

        y_train_full = y[train_idx]
        y_test = y[test_idx]

        result = train_one_fold(
            X_train_full,
            y_train_full,
            X_test,
            y_test,
            fold_num,
            seed,
            epochs=30
        )

        fold_results.append(result)

    eval_df = pd.DataFrame(fold_results)

    metric_cols = [
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "test_loss",
        "test_acc",
        "f1-score",
        "precision",
        "recall"
    ]

    avg_row = {"fold": "avg"}
    std_row = {"fold": "std"}

    for col in metric_cols:
        avg_row[col] = eval_df[col].mean()
        std_row[col] = eval_df[col].std()

    eval_df = pd.concat(
        [eval_df, pd.DataFrame([avg_row, std_row])],
        ignore_index=True
    )

    info_df = pd.DataFrame({
        "Variable": [
            "Seed",
            "Config",
            "Model",
            "DatasetPath",
            "NumSamples",
            "Channels",
            "TimePoints",
            "NumClasses",
            "Folds",
            "Epochs",
            "OuterSplit",
            "ValidationSplit"
        ],
        "Value": [
            seed,
            config_name,
            "RNN_LSTM_2class",
            data_path,
            X.shape[0],
            X.shape[1],
            X.shape[2],
            2,
            5,
            30,
            "5-fold StratifiedKFold",
            "20% of training fold"
        ]
    })

    save_path = os.path.join(save_folder, f"lstm_5fold_results_seed_{seed}.xlsx")

    with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
        info_df.to_excel(writer, sheet_name="info", index=False)
        eval_df.to_excel(writer, sheet_name="eval_history", index=False)

    print("\nSaved 5-fold results:")
    print(save_path)

# -----------------------------
# Main run: ICA and NoICA for all seeds
# -----------------------------
for seed in SEEDS:

    run_5fold(
        ica_file,
        os.path.join(result_dir, f"LSTM_2class_ICA_5fold_seed_{seed}"),
        "ICA",
        seed
    )

    run_5fold(
        noica_file,
        os.path.join(result_dir, f"LSTM_2class_NoICA_5fold_seed_{seed}"),
        "NoICA",
        seed
    )