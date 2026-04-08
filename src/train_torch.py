import os
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import numpy as np

from src.train import prepare_dataset, FEATURE_COLUMNS
from src.evaluate import evaluate_classifier


# ====== Model ======
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ====== Training ======
def train_model(X_train, y_train, epochs=20, lr=1e-3):
    device = torch.device("cpu")

    X = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32).to(device)

    model = MLP(X.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return model


# ====== Evaluation ======
def evaluate(model, X_test, y_test):
    model.eval()

    X = torch.tensor(X_test.values, dtype=torch.float32)
    with torch.no_grad():
        probs = model(X).numpy().flatten()

    preds = (probs > 0.5).astype(int)

    metrics = evaluate_classifier(y_test, preds, probs)
    return metrics


# ====== Save ======
def save_model(model, path="models/torch_model.pt"):
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), path)


# ====== Main ======
if __name__ == "__main__":
    X_train, y_train, X_test, y_test, _, _ = prepare_dataset()

    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print("Feature columns:", FEATURE_COLUMNS)

    model = train_model(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)

    print("\nTest metrics (PyTorch):")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    save_model(model)
    print("\nSaved PyTorch model to models/torch_model.pt")