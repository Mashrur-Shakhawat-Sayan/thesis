import flwr as fl
from flwr.server.strategy import FedAvg, FedOpt

def get_strategy(name="fedavg"):
    if name == "fedavg":
        return FedAvg()
    elif name == "fedopt":
        return FedOpt(learning_rate=0.01)
    else:
        raise ValueError("Unsupported strategy")

def main():
    import sys
    strategy = sys.argv[1] if len(sys.argv) > 1 else "fedavg"
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=get_strategy(strategy)
    )

if __name__ == "__main__":
    main()
