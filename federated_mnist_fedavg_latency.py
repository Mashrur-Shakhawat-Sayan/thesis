# federated_mnist_fedavg_latency.py
# ---------------------------------
# “Manual FedAvg” on MNIST with 10 clients,
# measuring latency (per‐client, server aggregation, total per round).

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

# 2) Split into 10 client shards
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
        tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])
    return model

# 4) One‐epoch client update function
def client_update(global_model, dataset, optimizer):
    local_model = create_keras_model()
    local_model.set_weights(global_model.get_weights())
    local_model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    local_model.fit(dataset, epochs=1, verbose=0)
    return local_model.get_weights()

# 5) FedAvg loop with latency measurement
NUM_ROUNDS = 20
global_model = create_keras_model()
global_model.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

for round_num in range(1, NUM_ROUNDS + 1):
    print(f"--- FedAvg Round {round_num} ---")
    round_start = time.perf_counter()

    client_times = []
    client_weights = []

    # (a) Each client trains locally for one epoch; measure timing
    for client_id in range(NUM_CLIENTS):
        c_start = time.perf_counter()
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.02)
        updated_w = client_update(global_model, client_datasets[client_id], optimizer)
        c_end = time.perf_counter()
        client_times.append(c_end - c_start)
        client_weights.append(updated_w)

    avg_client_time = sum(client_times) / NUM_CLIENTS

    # (b) Server aggregation (FedAvg): average updated weights; measure timing
    agg_start = time.perf_counter()
    new_weights = []
    for layer_idx in range(len(client_weights[0])):
        stack = np.stack([cw[layer_idx] for cw in client_weights], axis=0)
        new_weights.append(np.mean(stack, axis=0))
    global_model.set_weights(new_weights)
    agg_end = time.perf_counter()
    server_time = agg_end - agg_start

    # (c) Evaluate every 5 rounds (no timing)
    if round_num % 5 == 0:
        loss, acc = global_model.evaluate(test_dataset, verbose=0)
        print(f"  → Global test accuracy after round {round_num}: {acc:.4f}")

    round_end = time.perf_counter()
    round_time = round_end - round_start

    print(f"  Round {round_num} timings: "
          f"total = {round_time:.2f}s, "
          f"avg_client = {avg_client_time:.2f}s, "
          f"server = {server_time:.2f}s\n")

# 6) Save the final global model
global_model.save("fedavg_mnist_final.h5")
print("Training complete. Saved 'fedavg_mnist_final.h5'.")
