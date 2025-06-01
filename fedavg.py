import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score

# ============ DEVICE SETUP ============
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# ============ MODEL DEFINITION ============
class MNIST_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        if x.dim() == 4:
            x = x.view(-1, 28 * 28)
        elif x.dim() == 3:
            x = x.view(-1, 28 * 28)
        return self.fc2(F.relu(self.fc1(x)))

# ============ LOAD & SPLIT MNIST ============
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset  = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

NUM_CLIENTS = 100
CLIENT_SIZE = len(train_dataset) // NUM_CLIENTS  # 600 per client

# Create a DataLoader for each client’s 600 images
client_train_loaders = []
for i in range(NUM_CLIENTS):
    start  = i * CLIENT_SIZE
    end    = (i + 1) * CLIENT_SIZE
    subset = Subset(train_dataset, list(range(start, end)))
    loader = DataLoader(subset, batch_size=32, shuffle=True)
    client_train_loaders.append(loader)

# Single global test loader (everyone uses this to evaluate)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# ============ HELPER FUNCTIONS ============
def clone_model(model: nn.Module) -> nn.Module:
    cloned = MNIST_MLP().to(device)
    cloned.load_state_dict(model.state_dict())
    return cloned

def evaluate_accuracy_and_f1(model: nn.Module):
    model.eval()
    correct = 0
    total   = 0
    all_preds  = []
    all_labels = []

    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss   = criterion(logits, y)
            running_loss += loss.item() * X.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())

    avg_loss  = running_loss / total
    accuracy  = correct / total
    f1        = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, accuracy, f1

def model_size_bytes(model: nn.Module) -> int:
    total_params = sum(p.numel() for p in model.parameters())
    return total_params * 4  # float32 = 4 bytes each

# ============ FEDAVG PARAMETERS ============
NUM_ROUNDS      = 20
LOCAL_EPOCHS    = 1
CLIENT_FRACTION = 0.1    # 10% of 100 → 10 clients per round
CLIENT_LR       = 0.02
BATCH_SIZE      = 32

# 1) Initialize global model
global_model = MNIST_MLP().to(device)

print("\n=== Starting FedAvg Training ===")
for rnd in range(1, NUM_ROUNDS + 1):
    t0 = time.time()
    global_weights = global_model.state_dict()

    # 2a) Pick a random subset of clients
    num_sampled = max(int(CLIENT_FRACTION * NUM_CLIENTS), 1)
    selected_clients = np.random.choice(NUM_CLIENTS, num_sampled, replace=False)

    # 2b) Accumulate weighted sum of client models
    total_samples = 0
    accum_weights = [torch.zeros_like(w) for w in global_weights.values()]

    for cid in selected_clients:
        # 2b.i) Clone global → local
        local_model = MNIST_MLP().to(device)
        local_model.load_state_dict(global_weights)

        # 2b.ii) Train locally (SGD)
        optimizer = optim.SGD(local_model.parameters(), lr=CLIENT_LR)
        criterion = nn.CrossEntropyLoss()

        local_model.train()
        for _ in range(LOCAL_EPOCHS):
            for Xb, yb in client_train_loaders[cid]:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = local_model(Xb)
                loss   = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        # 2b.iii) Collect updated weights + sample count
        local_weights = local_model.state_dict()
        num_samples   = len(client_train_loaders[cid].dataset)

        # 2b.iv) Add weighted sum
        for i, key in enumerate(global_weights.keys()):
            accum_weights[i] += local_weights[key] * num_samples
        total_samples += num_samples

    # 2c) Compute new global weights = (1/total_samples) * accum_weights
    new_weights = {}
    for i, key in enumerate(global_weights.keys()):
        new_weights[key] = (accum_weights[i] / total_samples).cpu()
    global_model.load_state_dict(new_weights)

    # 2d) Evaluate on the global test set
    loss, accuracy, f1 = evaluate_accuracy_and_f1(global_model)
    t1 = time.time()
    round_time = t1 - t0
    size_mb = model_size_bytes(global_model) / 1e6

    print(f"Round {rnd:02d} | Time: {round_time:5.2f}s | "
          f"Loss: {loss:.4f} | Acc: {accuracy*100:5.2f}% | "
          f"F1: {f1:.4f} | Model size: {size_mb:.3f} MB")

print("=== FedAvg Training Complete ===")
