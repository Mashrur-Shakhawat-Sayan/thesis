import tensorflow as tf
import numpy as np
import psutil
import time
import os
from sklearn.metrics import precision_score, recall_score, f1_score

# ─── Helpers ────────────────────────────────────────────────────────────────────
def format_bytes(b: int) -> str:
    for unit in ('B','KB','MB','GB'):
        if b < 1024.0:
            return f"{b:0.2f}{unit}"
        b /= 1024.0
    return f"{b:.2f}TB"

def create_mlp_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28*28,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    return model

# ─── 1) Load & preprocess MNIST ────────────────────────────────────────────────
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype(np.float32) / 255.0
x_test = x_test.reshape(-1, 28*28).astype(np.float32) / 255.0
y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)

# ─── 2) Split into clients (Extreme Non-IID) ───────────────────────────────────
NUM_CLIENTS = 10
BATCH_SIZE = 64
digits_per_client = 10 // NUM_CLIENTS  # Number of classes per client

client_datasets = []
for i in range(NUM_CLIENTS):
    client_digits = [x for x in range(i * digits_per_client, (i + 1) * digits_per_client)]
    client_x = x_train[np.isin(y_train, client_digits)]
    client_y = y_train[np.isin(y_train, client_digits)]
    ds = (tf.data.Dataset
            .from_tensor_slices((client_x, client_y))
            .shuffle(len(client_x))
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE))
    client_datasets.append(ds)

# ─── 3) Initialize global model and server optimizer ────────────────────────────
global_model = create_mlp_model()
global_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
server_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)  # Lower server LR

# ─── 4) Metric tracking ────────────────────────────────────────────────────────
rounds = []
accs = []
losses = []
precisions = []
recalls = []
f1s = []
round_times = []
mem_changes = []
comm_overheads_mb = []
process = psutil.Process()

# ─── 5) Define client update function (Pure FedOpt, no FedProx) ────────────────
def client_update(global_model, dataset, lr=0.001, local_epochs=1):
    # Create a fresh model for this client
    client_model = create_mlp_model()
    client_model.set_weights(global_model.get_weights())
    
    # Compile with SGD optimizer
    client_model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Train on client data
    for epoch in range(local_epochs):
        for x_batch, y_batch in dataset:
            client_model.train_on_batch(x_batch, y_batch)
            
    return client_model.get_weights()

# ─── 6) Training loop with FedOpt ──────────────────────────────────────────────
NUM_ROUNDS = 40

# Calculate model size
temp_model = create_mlp_model()
temp_model.build(input_shape=(None, 28*28))
model_param_count = temp_model.count_params()
model_bytes = model_param_count * 4
model_size_mb = model_bytes / (1024**2)
del temp_model

for rnd in range(1, NUM_ROUNDS + 1):
    print(f"--- FedOpt Round {rnd} ---")
    mem0 = process.memory_info().rss
    t0 = time.time()

    # Store initial global weights
    global_weights = global_model.get_weights()
    
    # Client updates
    client_weights = []
    for ds in client_datasets:
        client_weights.append(client_update(global_model, ds))
    
    # FedOpt aggregation: compute pseudo-gradient
    avg_update = []
    for i in range(len(global_weights)):
        layer_updates = [cw[i] - global_weights[i] for cw in client_weights]
        avg_update.append(np.mean(layer_updates, axis=0))
    
    # Apply the update using the server optimizer
    # Create a list of gradients (negative of average update for optimizer)
    gradients = [-tf.convert_to_tensor(u) for u in avg_update]
    
    # Apply gradients with server optimizer
    server_optimizer.apply_gradients(zip(gradients, global_model.trainable_variables))

    # Evaluation
    test_loss, test_acc = global_model.evaluate(x_test, y_test, verbose=0)
    y_pred = np.argmax(global_model.predict(x_test, verbose=0), axis=1)
    
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    t1 = time.time()
    mem1 = process.memory_info().rss
    
    # Track metrics
    rounds.append(rnd)
    accs.append(test_acc)
    losses.append(test_loss)
    precisions.append(prec)
    recalls.append(rec)
    f1s.append(f1)
    round_times.append(t1 - t0)
    mem_changes.append(mem1 - mem0)
    comm_overheads_mb.append((2 * model_bytes * NUM_CLIENTS) / (1024**2))
    
    print(f"Acc={test_acc:.4f}, Loss={test_loss:.4f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}")
    print(f"Time: {t1-t0:.2f}s, Mem Δ={format_bytes(mem1-mem0)}, CommOv={comm_overheads_mb[-1]:.2f}MB")

# ─── 7) Save model ─────────────────────────────────────────────────────────────
global_model.save("fedopt_final_model.keras")