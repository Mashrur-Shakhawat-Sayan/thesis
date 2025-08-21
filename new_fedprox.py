import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # Disable TensorFlow 2.x behavior for compatibility
tf.compat.v1.enable_eager_execution()  # Enable eager execution for compatibility
import numpy as np
import psutil
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import time

# ─── Helpers ────────────────────────────────────────────────────────────────────
def format_bytes(b: int) -> str:
    for unit in ('B','KB','MB','GB'):
        if b < 1024.0:
            return f"{b:0.2f}{unit}"
        b /= 1024.0
    return f"{b:.2f}TB"

def create_mlp_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28*28,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),  # Add dropout after the first hidden layer
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),  # Add dropout after the second hidden layer
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

# ─── 1) Load & preprocess MNIST ────────────────────────────────────────────────
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype(np.float32) / 255.0
x_test  = x_test.reshape(-1, 28*28).astype(np.float32) / 255.0
y_train = y_train.astype(np.int64)
y_test  = y_test.astype(np.int64)

# ─── 2) Split into clients (Non-IID) ───────────────────────────────────────────
NUM_CLIENTS    = 10
BATCH_SIZE     = 64
digits_per_client = 10 // NUM_CLIENTS  # Number of classes per client

client_datasets = []
for i in range(NUM_CLIENTS):
    client_digits = [x for x in range(i * digits_per_client, (i + 1) * digits_per_client)]
    mask = np.isin(y_train, client_digits)
    client_x, client_y = x_train[mask], y_train[mask]
    if len(client_x) == 0:
        print(f"Skipping client {i} (no samples for digits {client_digits})")
        continue
    ds = (tf.data.Dataset
            .from_tensor_slices((client_x, client_y))
            .shuffle(buffer_size=max(1, len(client_x)))
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE))
    client_datasets.append(ds)

# ─── 3) Initialize global model ────────────────────────────────────────────────
global_model = create_mlp_model()
global_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ─── 4) Metric tracking ────────────────────────────────────────────────────────
rounds, accs, losses, precisions, recalls, f1s = [], [], [], [], [], []
avg_client_times, server_times, round_times, mem_changes = [], [], [], []
comm_overheads_mb, conv_rates, model_size_mb_hist = [], [], []

process = psutil.Process()

# ─── 5) FedProx client update function ─────────────────────────────────────────
def fedprox_client_update(global_model, dataset, mu=0.01, lr=0.01, local_epochs=4):
    """
    FedProx client update with proximal term to handle statistical heterogeneity
    mu: proximal term parameter (higher values enforce closer proximity to global model)
    """
    # Create local model and initialize with global weights
    local_model = create_mlp_model()
    local_model.set_weights(global_model.get_weights())
    
    # Get global model weights as tensors
    global_weights = [tf.constant(w) for w in global_model.get_weights()]
    
    # Use SGD optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    
    # Local training loop
    for epoch in range(local_epochs):
        for x_batch, y_batch in dataset:
            with tf.GradientTape() as tape:
                # Forward pass
                predictions = local_model(x_batch, training=True)
                
                # Calculate loss (cross-entropy + proximal term)
                ce_loss = tf.keras.losses.sparse_categorical_crossentropy(y_batch, predictions)
                ce_loss = tf.reduce_mean(ce_loss)
                
                # FedProx proximal term: mu/2 * ||w - w_global||^2
                prox_term = 0
                for w, w_g in zip(local_model.trainable_variables, global_weights):
                    prox_term += tf.reduce_sum(tf.square(w - w_g))
                prox_term = (mu / 2) * prox_term
                
                # Total loss
                total_loss = ce_loss + prox_term
            
            # Compute gradients and update local model
            gradients = tape.gradient(total_loss, local_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, local_model.trainable_variables))
    
    # Return the model update (difference between local and global weights)
    local_weights = local_model.get_weights()
    global_weights = global_model.get_weights()
    update = [local_w - global_w for local_w, global_w in zip(local_weights, global_weights)]
    
    return update

# ─── 6) FedProx Training Loop ──────────────────────────────────────────────────
NUM_ROUNDS = 80
MU = 0.1  # FedProx hyperparameter - controls the proximal term strength

# Calculate model size in MB
_temp = create_mlp_model()
_temp.build(input_shape=(None, 28*28))
model_size_mb_const = (_temp.count_params() * 4) / (1024**2)
del _temp

for rnd in range(1, NUM_ROUNDS + 1):
    print(f"--- FedProx Round {rnd} (μ={MU}) ---")
    mem0 = process.memory_info().rss
    t0 = time.perf_counter()

    # Store current global weights
    global_weights = global_model.get_weights()
    client_times = []
    updates = []

    # Client updates
    for i, ds in enumerate(client_datasets):
        c0 = time.perf_counter()
        update = fedprox_client_update(global_model, ds, mu=MU)
        c1 = time.perf_counter()
        client_times.append(c1 - c0)
        updates.append(update)
        print(f"  Client {i} completed in {c1-c0:.2f}s")

    avg_c = sum(client_times) / len(client_datasets)

    # Server aggregation (weighted average of updates)
    a0 = time.perf_counter()
    
    # Calculate number of samples per client for weighted averaging
    client_sizes = [len(list(ds.unbatch())) for ds in client_datasets]
    total_size = sum(client_sizes)
    
    # Apply weighted average of updates
    new_weights = []
    for i in range(len(global_weights)):
        layer_updates = np.zeros_like(global_weights[i])
        for j, update in enumerate(updates):
            layer_updates += update[i] * (client_sizes[j] / total_size)
        new_weights.append(global_weights[i] + layer_updates)
    
    # Update global model
    global_model.set_weights(new_weights)
    server_t = time.perf_counter() - a0

    # Evaluate global model
    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1000)
    loss, acc = global_model.evaluate(ds_test, verbose=0)
    probs = global_model.predict(x_test, batch_size=1000, verbose=0)
    preds = np.argmax(probs, axis=1)
    prec = precision_score(y_test, preds, average="macro", zero_division=0)
    rec = recall_score(y_test, preds, average="macro", zero_division=0)
    f1v = f1_score(y_test, preds, average="macro", zero_division=0)

    t1 = time.perf_counter()
    mem1 = process.memory_info().rss
    conv = 0.0 if len(losses) == 0 else abs(loss - losses[-1])

    # Store metrics
    rounds.append(rnd)
    accs.append(acc)
    losses.append(loss)
    precisions.append(prec)
    recalls.append(rec)
    f1s.append(f1v)
    avg_client_times.append(avg_c)
    server_times.append(server_t)
    round_times.append(t1 - t0)
    mem_changes.append(mem1 - mem0)
    comm_overheads_mb.append((2 * model_size_mb_const * NUM_CLIENTS))
    model_size_mb_hist.append(model_size_mb_const)
    conv_rates.append(conv)

    # Print summary
    print(f"Acc={acc:.4f}, Loss={loss:.4f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1v:.3f}, ConvRate={conv:.6f}")
    print(f"Times: avg_client={avg_c:.2f}s, server={server_t:.2f}s, round={(t1 - t0):.2f}s")
    print(f"Mem Δ={format_bytes(mem1 - mem0)}, CommOv={comm_overheads_mb[-1]:.2f}MB, ModelSize={model_size_mb_const:.2f}MB\n")

# ─── 7) Save final model and results ───────────────────────────────────────────
global_model.save("fedprox_final_model.keras")

# Plot results
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(rounds, accs, 'b-')
plt.title('Test Accuracy')
plt.xlabel('Rounds')
plt.ylabel('Accuracy')

plt.subplot(2, 2, 2)
plt.plot(rounds, losses, 'r-')
plt.title('Test Loss')
plt.xlabel('Rounds')
plt.ylabel('Loss')

plt.subplot(2, 2, 3)
plt.plot(rounds, [t/60 for t in round_times], 'g-')
plt.title('Round Time')
plt.xlabel('Rounds')
plt.ylabel('Time (minutes)')

plt.subplot(2, 2, 4)
plt.plot(rounds, [m/(1024*1024) for m in mem_changes], 'm-')
plt.title('Memory Change')
plt.xlabel('Rounds')
plt.ylabel('Memory (MB)')

plt.tight_layout()
plt.savefig('fedprox_results.png')
plt.show()

# Save metrics to file
metrics = {
    'rounds': rounds,
    'accuracies': accs,
    'losses': losses,
    'precisions': precisions,
    'recalls': recalls,
    'f1_scores': f1s,
    'avg_client_times': avg_client_times,
    'server_times': server_times,
    'round_times': round_times,
    'mem_changes': mem_changes,
    'comm_overheads': comm_overheads_mb,
    'conv_rates': conv_rates
}

np.savez('fedprox_metrics.npz', **metrics)
print("FedProx training completed. Model and metrics saved.")