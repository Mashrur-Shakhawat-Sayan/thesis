# simulate.py
import sys

import torch.optim as optim
import flwr as fl
from flwr.server.strategy import FedAvg, FedOpt
from flwr.common import ndarrays_to_parameters

from debas.client import FLClient  # your client implementation

NUM_CLIENTS = 600
NUM_ROUNDS = 20
STRATEGY = sys.argv[1] if len(sys.argv) > 1 else "fedavg"


def client_fn(cid: str):
    """Instantiate one of your FLClient objects (which inherits NumPyClient)."""
    return FLClient(cid=cid, num_clients=NUM_CLIENTS, strategy=STRATEGY).to_client()


def weighted_average(metrics: list[tuple[int, dict]]) -> dict:
    """Aggregates evaluate‐time metrics (accuracy)."""
    total_examples = sum(num_examples for num_examples, _ in metrics)
    # Weighted average of each client’s "accuracy"
    accuracy = sum(num_examples * m["accuracy"] for num_examples, m in metrics) / total_examples
    print(f"\n🌐 Global Accuracy This Round: {accuracy * 100:.2f}%")
    return {"accuracy": accuracy}


def fit_metrics_aggregator(metrics: list[tuple[int, dict]]) -> dict:
    """Aggregates fit‐time metrics (latency, memory)."""
    total_latency = sum(m.get("latency", 0.0) for _, m in metrics)
    total_memory = sum(m.get("memory", 0.0) for _, m in metrics)
    count = len(metrics)

    avg_latency = total_latency / count if count else 0.0
    avg_memory = total_memory / count if count else 0.0

    print(f"🕒 Avg Latency: {avg_latency:.4f}s | Total: {total_latency:.2f}s")
    print(f"📀 Avg Memory: {avg_memory:.2f}MB | Total: {total_memory:.2f}MB\n")

    return {
        "avg_latency": avg_latency,
        "total_latency": total_latency,
        "avg_memory": avg_memory,
        "total_memory": total_memory,
    }


def get_strategy():
    """
    Return a Flower strategy object based on STRATEGY.
    - For "fedavg": FedAvg with our custom aggregation functions.
    - For "fedopt": FedOpt with the correct keyword arguments (eta, eta_l, beta_1, beta_2, tau),
      plus an initial_parameters constructed from a dummy client’s weights.
    - For "fedprox": fall back to FedAvg (you can add any FedProx logic here if desired).
    """
    if STRATEGY == "fedavg":
        return FedAvg(
            evaluate_metrics_aggregation_fn=weighted_average,
            fit_metrics_aggregation_fn=fit_metrics_aggregator,
        )

    elif STRATEGY == "fedopt":
        # 1. Create a "dummy" client instance just to pull out its initial weights.
        dummy_client = FLClient(cid="0", num_clients=NUM_CLIENTS, strategy=STRATEGY)
        # 2. Call its get_parameters(...) method. Our FLClient.get_parameters ignores config
        #    and returns a list[numpy.ndarray]. Passing an empty dict is fine.
        initial_ndarrays = dummy_client.get_parameters(config={})
        # 3. Convert that list of NumPy ndarrays into a flwr.common.Parameters object:
        initial_parameters = ndarrays_to_parameters(initial_ndarrays)

        # 4. Now construct FedOpt using the signature in your Flower version. Based on
        #    the __init__ you pasted, we pass:
        #      - initial_parameters
        #      - eta, eta_l, beta_1, beta_2, tau
        #      - plus our aggregation functions
        return FedOpt(
            # How many clients to sample each round, etc.; you can tweak these if needed:
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=NUM_CLIENTS,
            min_evaluate_clients=NUM_CLIENTS,
            min_available_clients=NUM_CLIENTS,
            # Pass the initial model parameters we just extracted
            initial_parameters=initial_parameters,
            # Server‐side optimizer hyperparameters:
            eta=0.01,      # global learning rate
            eta_l=0.01,    # local learning rate
            beta_1=0.9,    # FedOpt’s β₁
            beta_2=0.999,  # FedOpt’s β₂
            tau=1e-9,      # FedOpt’s τ
            # Our custom aggregation callbacks:
            evaluate_metrics_aggregation_fn=weighted_average,
            fit_metrics_aggregation_fn=fit_metrics_aggregator,
        )

    elif STRATEGY == "fedprox":
        # Simply reuse FedAvg; you could insert a custom FedProx strategy here if you have one.
        return FedAvg(
            evaluate_metrics_aggregation_fn=weighted_average,
            fit_metrics_aggregation_fn=fit_metrics_aggregator,
        )

    else:
        raise ValueError(f"Unsupported strategy: {STRATEGY}")


def main():
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=get_strategy(),
    )


if __name__ == "__main__":
    main()
