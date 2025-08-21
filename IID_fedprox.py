import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # Disable TensorFlow 2.x behavior for compatibility
tf.compat.v1.enable_eager_execution()  # Enable eager execution for compatibility
import numpy as np
import psutil
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import time
import os

# ─── Helpers ────────────────────────────────────────────────────────────────────
def format_bytes(b: int) -> str:
    for unit in ('B','KB','MB','GB'):
        if b < 1024.0:
            return f"{b:0.2f}{unit}"
        b /= 1024.0
    return f"{b:.2f}TB"

def create_mlp_model(larger_model=True):
    """Create MLP model with option for larger architecture"""
    if larger_model:
        # Larger model with more capacity
        return tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(28*28,)),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])
    else:
        # Standard model
        return tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(28*28,)),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
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
client_sizes = []  # Store client dataset sizes for weighted averaging

for i in range(NUM_CLIENTS):
    client_digits = [x for x in range(i * digits_per_client, (i + 1) * digits_per_client)]
    mask = np.isin(y_train, client_digits)
    client_x, client_y = x_train[mask], y_train[mask]
    if len(client_x) == 0:
        print(f"Skipping client {i} (no samples for digits {client_digits})")
        continue
    client_sizes.append(len(client_x))
    ds = (tf.data.Dataset
            .from_tensor_slices((client_x, client_y))
            .shuffle(buffer_size=max(1, len(client_x)))
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE))
    client_datasets.append(ds)

# Calculate weights for weighted averaging
total_size = sum(client_sizes)
client_weights = [size / total_size for size in client_sizes]

# ─── 3) Initialize global model ────────────────────────────────────────────────
LARGER_MODEL = True  # Set to True to use larger model architecture
global_model = create_mlp_model(larger_model=LARGER_MODEL)
global_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ─── 4) Metric tracking ────────────────────────────────────────────────────────
rounds, accs, losses, precisions, recalls, f1s = [], [], [], [], [], []
avg_client_times, server_times, round_times, mem_changes = [], [], [], []
comm_overheads_mb, conv_rates, model_size_mb_hist = [], [], []
client_lrs = []  # Track client learning rates
mu_values = []   # Track mu values

process = psutil.Process()

# ─── 5) FedProx client update function ─────────────────────────────────────────
def fedprox_client_update(global_model, dataset, mu=0.01, lr=0.01, local_epochs=4):
    """
    FedProx client update with proximal term to handle statistical heterogeneity
    mu: proximal term parameter (higher values enforce closer proximity to global model)
    """
    # Create local model and initialize with global weights
    local_model = create_mlp_model(larger_model=LARGER_MODEL)
    local_model.set_weights(global_model.get_weights())
    
    # Get global model weights as tensors
    global_weights = [tf.constant(w) for w in global_model.get_weights()]
    
    # Use SGD optimizer with the provided learning rate
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    
    # Local training loop
    for epoch in range(local_epochs):
        for x_batch, y_batch in dataset:
            with tf.GradientTape() as tape:
                # Forward pass
                predictions = local_model(x_batch, training=True)
                
                # Calculate loss using sparse_softmax_cross_entropy
                ce_loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(
                    labels=y_batch, 
                    logits=predictions
                )
                
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
INITIAL_MU = 0.1  # Initial FedProx hyperparameter
INITIAL_CLIENT_LR = 0.02  # Initial client learning rate

# Adaptive mu - will increase every 10 rounds to combat client drift
current_mu = INITIAL_MU
current_client_lr = INITIAL_CLIENT_LR

# Calculate model size in MB
_temp = create_mlp_model(larger_model=LARGER_MODEL)
_temp.build(input_shape=(None, 28*28))
model_size_mb_const = (_temp.count_params() * 4) / (1024**2)
del _temp

# Create directory for saving checkpoints
os.makedirs("checkpoints", exist_ok=True)

for rnd in range(1, NUM_ROUNDS + 1):
    # Adaptive hyperparameter tuning
    if rnd % 10 == 0 and rnd > 0:
        # Increase mu to combat client drift in later rounds
        current_mu = min(0.5, current_mu * 1.2)
        # Decay learning rate for finer tuning
        current_client_lr = max(0.005, current_client_lr * 0.9)
        print(f"Adjusted hyperparameters: μ={current_mu:.3f}, LR={current_client_lr:.4f}")
    
    print(f"--- FedProx Round {rnd} (μ={current_mu:.3f}, LR={current_client_lr:.4f}) ---")
    mem0 = process.memory_info().rss
    t0 = time.perf_counter()

    # Store current global weights
    global_weights = global_model.get_weights()
    client_times = []
    updates = []

    # Client updates
    for i, ds in enumerate(client_datasets):
        c0 = time.perf_counter()
        update = fedprox_client_update(global_model, ds, mu=current_mu, lr=current_client_lr)
        c1 = time.perf_counter()
        client_times.append(c1 - c0)
        updates.append(update)
        if rnd % 10 == 0:  # Only print client times occasionally to reduce output
            print(f"  Client {i} completed in {c1-c0:.2f}s")

    avg_c = sum(client_times) / len(client_datasets)

    # Server aggregation (weighted average of updates)
    a0 = time.perf_counter()
    
    # Apply weighted average of updates using precomputed weights
    new_weights = []
    for i in range(len(global_weights)):
        layer_updates = np.zeros_like(global_weights[i])
        for j, update in enumerate(updates):
            layer_updates += update[i] * client_weights[j]
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
    client_lrs.append(current_client_lr)
    mu_values.append(current_mu)

    # Print summary
    print(f"Acc={acc:.4f}, Loss={loss:.4f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1v:.3f}, ConvRate={conv:.6f}")
    print(f"Times: avg_client={avg_c:.2f}s, server={server_t:.2f}s, round={(t1 - t0):.2f}s")
    print(f"Mem Δ={format_bytes(mem1 - mem0)}, CommOv={comm_overheads_mb[-1]:.2f}MB, ModelSize={model_size_mb_const:.2f}MB\n")
    
    # Save checkpoint every 10 rounds
    if rnd % 10 == 0:
        checkpoint_path = f"checkpoints/fedprox_checkpoint_round_{rnd}.keras"
        global_model.save(checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

# ─── 7) Save final model and results ───────────────────────────────────────────
global_model.save("fedprox_final_model.keras")

# Enhanced plotting
plt.figure(figsize=(15, 10))

# Accuracy plot
plt.subplot(2, 3, 1)
plt.plot(rounds, accs, 'b-')
plt.title('Test Accuracy')
plt.xlabel('Rounds')
plt.ylabel('Accuracy')
plt.grid(True)

# Loss plot
plt.subplot(2, 3, 2)
plt.plot(rounds, losses, 'r-')
plt.title('Test Loss')
plt.xlabel('Rounds')
plt.ylabel('Loss')
plt.grid(True)

# Learning rate and mu plot
plt.subplot(2, 3, 3)
plt.plot(rounds, client_lrs, 'g-', label='Client LR')
plt.plot(rounds, mu_values, 'm-', label='μ')
plt.title('Hyperparameters')
plt.xlabel('Rounds')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# Round time plot
plt.subplot(2, 3, 4)
plt.plot(rounds, [t/60 for t in round_times], 'c-')
plt.title('Round Time')
plt.xlabel('Rounds')
plt.ylabel('Time (minutes)')
plt.grid(True)

# Memory usage plot
plt.subplot(2, 3, 5)
plt.plot(rounds, [m/(1024*1024) for m in mem_changes], 'y-')
plt.title('Memory Change')
plt.xlabel('Rounds')
plt.ylabel('Memory (MB)')
plt.grid(True)

# Convergence rate plot
plt.subplot(2, 3, 6)
plt.plot(rounds, conv_rates, 'k-')
plt.title('Convergence Rate')
plt.xlabel('Rounds')
plt.ylabel('|ΔLoss|')
plt.grid(True)

plt.tight_layout()
plt.savefig('fedprox_results.png', dpi=300, bbox_inches='tight')
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
    'conv_rates': conv_rates,
    'client_learning_rates': client_lrs,
    'mu_values': mu_values
}

np.savez('fedprox_metrics.npz', **metrics)
print("FedProx training completed. Model and metrics saved.")

# Print final performance summary
print("\n=== TRAINING SUMMARY ===")
print(f"Final Accuracy: {accs[-1]:.4f}")
print(f"Final Loss: {losses[-1]:.4f}")
print(f"Best Accuracy: {max(accs):.4f} (Round {np.argmax(accs)+1})")
print(f"Total Training Time: {sum(round_times)/60:.2f} minutes")