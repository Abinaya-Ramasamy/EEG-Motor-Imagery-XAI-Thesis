import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

SEED = 87

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_PATH = "../dataset/bci2a_2class_autoICA_all_subjects.npz"
RESULT_DIR = "../result/EEGNet_ICA_FIXED_FOR_XAI"
os.makedirs(RESULT_DIR, exist_ok=True)

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

# Load data
data = np.load(DATASET_PATH)
X = data["X"]
y = data["y"]

print("X:", X.shape)
print("y:", y.shape)
print("Class counts:", np.bincount(y.astype(int)))

# Fixed split for XAI
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

# Normalize using train mean/std
mean = X_train.mean()
std = X_train.std()

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Save exact test split for XAI
np.savez(
    os.path.join(RESULT_DIR, "xai_fixed_test_data.npz"),
    X_test=X_test,
    y_test=y_test
)

# Reshape for EEGNet
X_train = X_train[:, np.newaxis, :, :]
X_test = X_test[:, np.newaxis, :, :]

X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t = torch.tensor(y_test, dtype=torch.long)

g = torch.Generator()
g.manual_seed(SEED)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=32,
    shuffle=True,
    generator=g
)

test_loader = DataLoader(
    TensorDataset(X_test_t, y_test_t),
    batch_size=32,
    shuffle=False
)

model = EEGNet().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 30

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for Xb, yb in train_loader:
        Xb = Xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(Xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}, Train Acc: {correct/total:.4f}")

# Evaluate
model.eval()
all_preds = []
all_true = []

with torch.no_grad():
    for Xb, yb in test_loader:
        Xb = Xb.to(DEVICE)
        outputs = model(Xb)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_true.extend(yb.numpy())

acc = accuracy_score(all_true, all_preds)
report = classification_report(all_true, all_preds, digits=4)
cm = confusion_matrix(all_true, all_preds)

print("\nTest Accuracy:", acc)
print(report)
print(cm)

print("Prediction counts:", np.bincount(np.array(all_preds).astype(int)))
print("Correct label 0:", np.sum((np.array(all_true) == 0) & (np.array(all_preds) == np.array(all_true))))
print("Correct label 1:", np.sum((np.array(all_true) == 1) & (np.array(all_preds) == np.array(all_true))))

# Save model
torch.save(model.state_dict(), os.path.join(RESULT_DIR, "model.pth"))

# Save predictions
np.savez(
    os.path.join(RESULT_DIR, "xai_fixed_predictions.npz"),
    y_test=np.array(all_true),
    preds=np.array(all_preds)
)

# Save Excel result
report_df = pd.DataFrame(classification_report(all_true, all_preds, output_dict=True)).transpose()
cm_df = pd.DataFrame(cm)
acc_df = pd.DataFrame({"Metric": ["Accuracy"], "Value": [acc]})

with pd.ExcelWriter(os.path.join(RESULT_DIR, "training_result.xlsx")) as writer:
    acc_df.to_excel(writer, sheet_name="Accuracy", index=False)
    report_df.to_excel(writer, sheet_name="Classification Report")
    cm_df.to_excel(writer, sheet_name="Confusion Matrix", index=False)

print("\nSaved fixed XAI model and test data to:", RESULT_DIR)