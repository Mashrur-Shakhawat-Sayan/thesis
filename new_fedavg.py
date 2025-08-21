import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # Disable TensorFlow 2.x behavior for compatibility
tf.compat.v1.enable_eager_execution()  # Enable eager execution for compatibility
import numpy as np
import psutil
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import time  # Ensure the 'time' module is imported

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
CLIENT_SZ      = len(x_train) // NUM_CLIENTS
BATCH_SIZE     = 64

client_datasets = []
digits_per_client = 10 // NUM_CLIENTS  # Number of classes per client

for i in range(NUM_CLIENTS):
    client_digits = [x for x in range(i * digits_per_client, (i + 1) * digits_per_client)]
    client_x = x_train[np.isin(y_train, client_digits)]  # Ensure each client gets unique classes
    client_y = y_train[np.isin(y_train, client_digits)]
    ds = (tf.data.Dataset
            .from_tensor_slices((client_x, client_y))
            .shuffle(len(client_x))
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE))
    client_datasets.append(ds)

# ─── 3) Initialize global model ────────────────────────────────────────────────
global_model = create_mlp_model()
global_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ─── 4) Metric tracking ─────────────────────────────────────────────────────────
rounds            = []
accs              = []
losses            = []
precisions        = []
recalls           = []
f1s               = []
avg_client_times  = []
server_times      = []
round_times       = []
mem_changes       = []
comm_overheads_mb = []
conv_rates        = []
model_size_mb_hist  = []
model_size_mb_const = 0  # Initialize model size here
process = psutil.Process()

# ─── 5) Define client update function ───────────────────────────────────────────
def client_update(global_model, dataset, mu=0.01, lr=0.001, local_epochs=1):
    local = create_mlp_model()
    local.set_weights(global_model.get_weights())  # Set initial weights from global model
    # Use SGD with momentum instead of Adam
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)  # SGD with momentum
    train_vars = local.trainable_variables
    global_tensors = [tf.convert_to_tensor(w, dtype=tf.float32) for w in global_model.get_weights()]

    @tf.function
    def step(xb, yb):
        with tf.GradientTape() as tape:
            logits = local(xb, training=True)
            ce_loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(yb, logits)
            )
            # Add FedProx (proximal term)
            prox_term = tf.add_n([tf.reduce_sum(tf.square(var - gw)) for var, gw in zip(train_vars, global_tensors)]) * (mu / 2.0)
            loss = ce_loss + prox_term  # Total loss (cross-entropy + proximal term)
        grads = tape.gradient(loss, train_vars)
        optimizer.apply_gradients(zip(grads, train_vars))

    for _ in range(local_epochs):
        for xb, yb in dataset:
            step(xb, yb)

    return local.get_weights()  # Return updated weights from the client model

# ─── 6) Training loop ───────────────────────────────────────────────────────────
NUM_ROUNDS = 40

# Calculate model size in bytes
_temp = create_mlp_model()
_temp.build(input_shape=(None, 28*28))  # Initialize the model
model_param_count = _temp.count_params()
model_bytes = model_param_count * 4  # Model size in bytes (since each parameter is a float32, which is 4 bytes)
del _temp  # Clean up the temporary model

model_size_mb_const = (model_bytes) / (1024**2)  # Convert to MB

for rnd in range(1, NUM_ROUNDS+1):
    print(f"--- FedAvg Round {rnd} ---")
    mem0 = process.memory_info().rss
    t0   = time.perf_counter()

    gw = global_model.get_weights()
    client_times = []
    updates      = []

    for ds in client_datasets:
        c0    = time.perf_counter()
        w_upd = client_update(global_model, ds)
        c1    = time.perf_counter()
        client_times.append(c1 - c0)
        updates.append(w_upd)

    avg_c = sum(client_times) / NUM_CLIENTS

    # Average the client updates
    a0 = time.perf_counter()
    new_weights = []
    for layer in range(len(gw)):
        stack = np.stack([upd[layer] for upd in updates], axis=0)
        new_weights.append(np.mean(stack, axis=0))
    global_model.set_weights(new_weights)
    server_t = time.perf_counter() - a0

    # Evaluate model performance
    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1000)
    loss, acc = global_model.evaluate(ds_test, verbose=0)
    probs = global_model.predict(x_test, batch_size=1000, verbose=0)
    preds = np.argmax(probs, axis=1)
    prec  = precision_score(y_test, preds, average="macro", zero_division=0)
    rec   = recall_score(y_test, preds, average="macro", zero_division=0)
    f1v   = f1_score(y_test, preds, average="macro", zero_division=0)

    t1   = time.perf_counter()
    mem1 = process.memory_info().rss

    conv = 0.0 if len(losses) == 0 else abs(loss - losses[-1])
    conv_rates.append(conv)

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
    comm_overheads_mb.append((2 * model_bytes * NUM_CLIENTS) / (1024**2))
    model_size_mb_hist.append(model_size_mb_const)

    # 🔹 Print round summary
    print(f"Acc={acc:.4f}, Loss={loss:.4f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1v:.3f}, ConvRate={conv:.6f}")
    print(f"Times: avg_client={avg_c:.2f}s, server={server_t:.2f}s, round={(t1-t0):.2f}s")
    print(f"Mem Δ={format_bytes(mem1-mem0)}, CommOv={comm_overheads_mb[-1]:.2f}MB, ModelSize={model_size_mb_const:.2f}MB\n")

# ─── 7) Save model ─────────────────────────────────────────────────────────────
global_model.save("fedavg__final_model.keras")
