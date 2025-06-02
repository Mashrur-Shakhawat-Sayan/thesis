import flwr as fl
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import psutil, time
from debas.model import MNISTModel
from torch.utils.data import random_split
import gc

BATCH_SIZE = 32

# Dataset
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)



def get_dataloader(partition: int, num_partitions: int, train=True):
    dataset = trainset
    length = len(dataset) // num_partitions
    start = partition * length
    end = start + length
    client_subset = torch.utils.data.Subset(dataset, list(range(start, end)))

    # Split into 75% train, 25% test
    train_len = int(0.75 * len(client_subset))
    test_len = len(client_subset) - train_len
    train_subset, test_subset = random_split(client_subset, [train_len, test_len])

    if train:
        return torch.utils.data.DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    else:
        return torch.utils.data.DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)


# Flower Client
class FLClient(fl.client.NumPyClient):
    def __init__(self, cid, num_clients, strategy="fedavg"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cid = int(cid)
        self.model = MNISTModel()
        self.num_clients = num_clients
        self.strategy = strategy

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        trainloader = get_dataloader(self.cid, self.num_clients, train=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        if self.strategy == "fedprox":
            mu = 0.01
            global_params = [torch.tensor(p) for p in parameters]

        self.model.train()
        start_time = time.time()
        total_loss = 0.0  # Track total loss

        for epoch in range(1):
            for x, y in trainloader:
                optimizer.zero_grad()
                pred = self.model(x)
                loss = F.cross_entropy(pred, y)
                if self.strategy == "fedprox":
                    prox_term = 0.0
                    for param, g_param in zip(self.model.parameters(), global_params):
                        prox_term += ((param - g_param) ** 2).sum()
                    loss += (mu / 2) * prox_term
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        latency = time.time() - start_time
        memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        avg_train_loss = total_loss / len(trainloader)

        if self.cid == "0":
            print(f"[Sample Client {self.cid}] Latency: {latency:.2f}s | Memory: {memory:.2f}MB | Train Loss: {avg_train_loss:.4f}")
        
        torch.cuda.empty_cache()
        gc.collect()
        del x, y, pred, loss
        if self.strategy == "fedprox":
            del global_params

        return self.get_parameters({}), len(trainloader.dataset), {
            "loss": avg_train_loss,
            "latency": latency,
            "memory": memory
        }


    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.to(self.device)
        self.model.eval()

        testloader = get_dataloader(self.cid, self.num_clients, train=False)
        criterion = torch.nn.CrossEntropyLoss()
        test_loss = 0.0
        correct, total = 0, 0

        start_time = time.time()
        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                loss = criterion(outputs, y)
                test_loss += loss.item()
                pred = outputs.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        latency = time.time() - start_time
        memory = psutil.Process().memory_info().rss / 1024 / 1024
        accuracy = correct / total
        avg_loss = test_loss / len(testloader)

        return float(avg_loss), total, {
            "accuracy": accuracy,
            "loss": avg_loss,
            "latency": latency,
            "memory": memory
        }


def main():
    import sys
    cid = sys.argv[1]
    num_clients = 10
    strategy = sys.argv[2] if len(sys.argv) > 2 else "fedavg"
    fl.client.start_numpy_client("localhost:8080", client=FLClient(cid, num_clients, strategy))

if __name__ == "__main__":
    main()
