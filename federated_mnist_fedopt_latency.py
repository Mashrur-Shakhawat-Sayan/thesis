# federated_mnist_fedopt_latency.py
# ----------------------------------
# Manual “FedOpt” on MNIST with 10 clients, measuring latency:
#  - average time each client spends in local training
#  - time server spends aggregating + applying Adam
#  - total time per round

import time
import numpy as np
import tensorflow as tf

# 1) Load & preprocess MNIST
(x_train_all, y_train_all), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train_all = x_train_all.astype(np.float32) / 255.0
x_test      = x_test.astype(np.float32)      / 255.0
y_train_all = y_train_all.astype(np.int64)
y_test      = y_test.astype(np.int64)

print("MNIST loaded: train =", x_train_all.shape, y_train_all.shape,
      "| test =", x_test.shape, y_test.shape)

# 2) Split into 10 clients (6,000 examples each)
NUM_CLIENTS = 10
CLIENT_DATA_SIZE = len(x_train_all) // NUM_CLIENTS  # 6,000 per client
BATCH_SIZE = 20

client_datasets = []
for i in range(NUM_CLIENTS):
    start = i * CLIENT_DATA_SIZE
    end   = (i + 1) * CLIENT_DATA_SIZE
    x_part = x_train_all[start:end]
    y_part = y_train_all[start:end]
    ds = tf.data.Dataset.from_tensor_slices((x_part, y_part)) \
                       .shuffle(CLIENT_DATA_SIZE) \
                       .batch(BATCH_SIZE)
    client_datasets.append(ds)

# Central test set, batched
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1000)
print(f"Created {NUM_CLIENTS} client datasets, each ~{CLIENT_DATA_SIZE} examples.\n")

# 3) Helper to build a fresh Keras CNN model
def create_keras_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28)),
        tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, kernel_size=3, activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    return model

# 4) One‐epoch client update (SGD(0.02))
def client_update(global_model, dataset, optimizer):
    local_model = create_keras_model()
    local_model.set_weights(global_model.get_weights())
    local_model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    local_model.fit(dataset, epochs=1, verbose=0)
    return local_model.get_weights()

# 5) Build & initialize the “server model” + server Adam optimizer
global_model = create_keras_model()
global_model.compile(
    optimizer="sgd",  # placeholder
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Use Adam on the server; its "gradient" will be -avg_delta
server_optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

def server_opt_step(global_vars, avg_delta):
    """
    Applies one Adam step on the server, treating -avg_delta as the
    gradient. That implements w_new = w_old + alpha * avg_delta (with Adam momentum).
    """
    grads_and_vars = []
    for var, d in zip(global_vars, avg_delta):
        # feed minus‐delta so that Adam does: var = var - lr*(-delta) = var + lr*delta
        grad_tensor = tf.convert_to_tensor(-d, dtype=var.dtype)
        grads_and_vars.append((grad_tensor, var))
    server_optimizer.apply_gradients(grads_and_vars)

# 6) FedOpt Training Loop with Latency Measurement
NUM_ROUNDS = 20

for round_num in range(1, NUM_ROUNDS + 1):
    print(f"--- FedOpt Round {round_num} ---")
    round_start = time.perf_counter()

    # (a) Each client trains locally; measure per-client time
    global_weights = global_model.get_weights()
    client_deltas = []
    client_times = []

    for client_id in range(NUM_CLIENTS):
        c_start = time.perf_counter()
        updated_weights = client_update(
            global_model,
            client_datasets[client_id],
            tf.keras.optimizers.SGD(learning_rate=0.02)
        )
        c_end = time.perf_counter()
        client_times.append(c_end - c_start)

        # Compute delta_i = updated_weights - global_weights
        delta_i = [u - g for u, g in zip(updated_weights, global_weights)]
        client_deltas.append(delta_i)

    avg_client_time = sum(client_times) / NUM_CLIENTS

    # (b) Average those deltas
    avg_delta = []
    for layer_idx in range(len(global_weights)):
        layer_stack = np.stack(
            [client_deltas[i][layer_idx] for i in range(NUM_CLIENTS)],
            axis=0
        )
        avg_layer_delta = np.mean(layer_stack, axis=0)
        avg_delta.append(avg_layer_delta)

    # (c) Server aggregation/Adam step; measure its latency
    agg_start = time.perf_counter()
    server_opt_step(global_model.trainable_variables, avg_delta)
    agg_end = time.perf_counter()
    server_time = agg_end - agg_start

    # (d) Evaluate every 5 rounds (no need to time this, but you can)
    if round_num % 5 == 0:
        loss, acc = global_model.evaluate(test_dataset, verbose=0)
        print(f"  → Global test accuracy after round {round_num}: {acc:.4f}")

    round_end = time.perf_counter()
    round_time = round_end - round_start

    print(f"  Round {round_num} timings: "
          f"  total = {round_time:.2f}s, "
          f"avg_client = {avg_client_time:.2f}s, "
          f"server = {server_time:.2f}s\n")

# 7) Save final global model
global_model.save("fedopt_mnist_final.h5")
print("FedOpt training complete. Saved 'fedopt_mnist_final.h5'.")
