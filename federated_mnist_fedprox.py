# federated_mnist_fedprox_latency.py
# ----------------------------------
# “Manual FedProx” simulation on MNIST with 100 clients,
# adding a proximal term (μ/2 ‖w - w_global‖^2) to each client’s loss,
# and measuring latency (per‐client, server aggregation, total per round).

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

# 2) Split into 100 clients (600 examples each)
NUM_CLIENTS = 100
CLIENT_DATA_SIZE = len(x_train_all) // NUM_CLIENTS  # 600 per client
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

# 4) Client update with proximal term μ (μ>0 means FedProx; μ=0 reduces to FedAvg)
def client_update(global_weights, dataset, mu, client_optimizer):
    """
    global_weights: list of numpy arrays = the global model's weights at round t
    dataset:        tf.data.Dataset for this client
    mu:             proximal coefficient (float)
    client_optimizer: an uncompiled optimizer (e.g. SGD(0.02))

    Returns: updated_weights (list of numpy arrays) after one local epoch.
    """
    # Build a new model and set its weights = global_weights
    local_model = create_keras_model()
    local_model.set_weights(global_weights)

    # Convert global_weights to tf tensors
    global_tensors = [tf.convert_to_tensor(w, dtype=tf.float32) for w in global_weights]
    train_vars = local_model.trainable_variables

    # Single‐epoch training loop with custom loss
    for (x_batch, y_batch) in dataset:
        with tf.GradientTape() as tape:
            # Standard cross‐entropy loss
            logits = local_model(x_batch, training=True)
            ce_loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(y_batch, logits)
            )
            # Proximal term: (mu/2) * Σ_j || w_j - w_global_j ||^2
            prox_term = tf.add_n([
                tf.reduce_sum(tf.square(var - gw))
                for var, gw in zip(train_vars, global_tensors)
            ])
            prox_term = (mu / 2.0) * prox_term

            # Total loss = CE loss + proximal term
            total_loss = ce_loss + prox_term

        grads = tape.gradient(total_loss, train_vars)
        client_optimizer.apply_gradients(zip(grads, train_vars))

    return local_model.get_weights()

# 5) FedProx training loop
NUM_ROUNDS = 20
MU = 0.01  # proximal coefficient

# Initialize the global model
global_model = create_keras_model()
global_model.compile(
    optimizer="sgd",  # placeholder; we do manual updates
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

for round_num in range(1, NUM_ROUNDS + 1):
    print(f"--- FedProx Round {round_num} (μ={MU}) ---")
    round_start = time.perf_counter()

    current_global_weights = global_model.get_weights()
    client_times = []
    client_weights = []

    # Each client does one epoch with the proximal term; measure timing
    for client_id in range(NUM_CLIENTS):
        c_start = time.perf_counter()
        client_optimizer = tf.keras.optimizers.SGD(learning_rate=0.02)
        updated_w = client_update(current_global_weights, client_datasets[client_id], MU, client_optimizer)
        c_end = time.perf_counter()
        client_times.append(c_end - c_start)
        client_weights.append(updated_w)

    avg_client_time = sum(client_times) / NUM_CLIENTS

    # Server aggregation: average updated weights; measure timing
    agg_start = time.perf_counter()
    new_global_weights = []
    for layer_idx in range(len(current_global_weights)):
        layer_stack = np.stack([client_weights[i][layer_idx] for i in range(NUM_CLIENTS)], axis=0)
        new_global_weights.append(np.mean(layer_stack, axis=0))
    global_model.set_weights(new_global_weights)
    agg_end = time.perf_counter()
    server_time = agg_end - agg_start

    # Evaluate every 5 rounds
    if round_num % 5 == 0:
        loss, acc = global_model.evaluate(test_dataset, verbose=0)
        print(f"  → Global test accuracy after round {round_num}: {acc:.4f}")

    round_end = time.perf_counter()
    round_time = round_end - round_start

    print(f"  Round {round_num} timings: "
          f"total = {round_time:.2f}s, "
          f"avg_client = {avg_client_time:.2f}s, "
          f"server = {server_time:.2f}s\n")

# 6) Save final model
global_model.save("fedprox_mnist_final.h5")
print("FedProx training complete. Saved 'fedprox_mnist_final.h5'.")
