import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load & Preprocess 
(ds_train_raw, ds_test_raw), _ = tfds.load(
    'mnist', split=['train', 'test'], as_supervised=True, with_info=True
)

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [-1])
    label = tf.one_hot(label, depth=10)
    return image, label

#  He Initialization
def he_init(shape):
    return tf.Variable(tf.random.normal(shape, stddev=tf.sqrt(2.0 / shape[0])))

# Build Network
def build_network(hidden_size):
    W1 = he_init([784, hidden_size])
    b1 = tf.Variable(tf.zeros([hidden_size]))
    W2 = he_init([hidden_size, hidden_size // 2])
    b2 = tf.Variable(tf.zeros([hidden_size // 2]))
    W3 = he_init([hidden_size // 2, 10])
    b3 = tf.Variable(tf.zeros([10]))
    return [W1, b1, W2, b2, W3, b3]

# Forward Pass 
def forward_pass(X, params, activation='relu'):
    W1, b1, W2, b2, W3, b3 = params
    act_fn = {'relu': tf.nn.relu, 'sigmoid': tf.nn.sigmoid, 'tanh': tf.nn.tanh}[activation]
    A1 = act_fn(tf.matmul(X, W1) + b1)
    A2 = act_fn(tf.matmul(A1, W2) + b2)
    return tf.matmul(A2, W3) + b3

#  Training Function 
def train_and_evaluate(activation='relu', hidden_size=256, lr=0.01,
                        batch_size=128, epochs=10):
    params = build_network(hidden_size)

    ds_train = ds_train_raw.map(preprocess).shuffle(60000).batch(batch_size)
    ds_test  = ds_test_raw.map(preprocess).batch(batch_size)

    steps = 60000 // batch_size
    train_acc_history, train_loss_history = [], []

    for epoch in range(epochs):
        correct = total = 0
        epoch_loss = 0.0

        for X_batch, y_batch in ds_train.take(steps):
            with tf.GradientTape() as tape:
                logits = forward_pass(X_batch, params, activation)
                loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=y_batch, logits=logits))
            grads = tape.gradient(loss, params)
            for p, g in zip(params, grads):
                p.assign_sub(lr * g)

            epoch_loss += loss.numpy()
            preds  = tf.argmax(logits, axis=1)
            labels = tf.argmax(y_batch, axis=1)
            correct += tf.reduce_sum(tf.cast(preds == labels, tf.float32)).numpy()
            total   += X_batch.shape[0]

        avg_loss = epoch_loss / steps
        train_acc_history.append(correct / total)
        train_loss_history.append(avg_loss)
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {correct/total:.4f}")

    # Test accuracy + collect predictions
    correct = total = 0
    all_preds, all_labels = [], []

    for X_batch, y_batch in ds_test:
        logits = forward_pass(X_batch, params, activation)
        preds  = tf.argmax(logits, axis=1)
        labels = tf.argmax(y_batch, axis=1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())
        correct += tf.reduce_sum(tf.cast(preds == labels, tf.float32)).numpy()
        total   += X_batch.shape[0]

    test_acc = correct / total
    return test_acc, train_acc_history, train_loss_history, all_preds, all_labels


# Helper: plot accuracy curves + confusion matrices 
def plot_experiment(title, param_name, param_values, results):
    n = len(results)

    # Accuracy curves
    plt.figure(figsize=(4*n, 4))
    for i, (label, test_acc, acc_hist, _, _, _) in enumerate(results):
        plt.subplot(1, n, i+1)
        plt.plot(acc_hist)
        plt.title(f"{param_name}={label}\nTest Acc: {test_acc:.4f}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(0.7, 1.0)
    plt.tight_layout()
    plt.suptitle(f"Effect of {title} — Accuracy Curves", y=1.02)
    plt.show()

    # Confusion matrices
    plt.figure(figsize=(5*n, 4))
    for i, (label, test_acc, _, _, all_preds, all_labels) in enumerate(results):
        cm = confusion_matrix(all_labels, all_preds)
        plt.subplot(1, n, i+1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=range(10), yticklabels=range(10),
                    cbar=False, annot_kws={"size": 6})
        plt.title(f"{param_name}={label}\nTest Acc: {test_acc:.4f}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
    plt.tight_layout()
    plt.suptitle(f"Effect of {title} — Confusion Matrices", y=1.02)
    plt.show()

# EXPERIMENT A: Vary Activation Function
print("\n===== Varying Activation Function =====")
activations = ['relu', 'sigmoid', 'tanh']
act_results = {}
act_data = []

for act in activations:
    print(f"\nActivation: {act}")
    test_acc, acc_hist, loss_hist, preds, labels = train_and_evaluate(activation=act, epochs=10)
    act_results[act] = test_acc
    act_data.append((act, test_acc, acc_hist, loss_hist, preds, labels))
    print(f"  --> Test Accuracy: {test_acc:.4f}")

plot_experiment("Activation Function", "Activation", activations, act_data)

# EXPERIMENT B: Vary Hidden Layer Size
print("\n===== Varying Hidden Layer Size =====")
hidden_sizes = [64, 256, 512]
size_results = {}
size_data = []

for hs in hidden_sizes:
    print(f"\nHidden Size: {hs}")
    test_acc, acc_hist, loss_hist, preds, labels = train_and_evaluate(hidden_size=hs, epochs=10)
    size_results[hs] = test_acc
    size_data.append((str(hs), test_acc, acc_hist, loss_hist, preds, labels))
    print(f"  --> Test Accuracy: {test_acc:.4f}")

plot_experiment("Hidden Layer Size", "Hidden", hidden_sizes, size_data)

# EXPERIMENT C: Vary Learning Rate
print("\n===== Varying Learning Rate =====")
learning_rates = [0.001, 0.01, 0.1]
lr_results = {}
lr_data = []

for lr in learning_rates:
    print(f"\nLearning Rate: {lr}")
    test_acc, acc_hist, loss_hist, preds, labels = train_and_evaluate(lr=lr, epochs=10)
    lr_results[lr] = test_acc
    lr_data.append((str(lr), test_acc, acc_hist, loss_hist, preds, labels))
    print(f"  --> Test Accuracy: {test_acc:.4f}")

# Loss curves for LR
n = len(lr_data)
plt.figure(figsize=(4*n, 4))
for i, (label, test_acc, _, loss_hist, _, _) in enumerate(lr_data):
    plt.subplot(1, n, i+1)
    plt.plot(loss_hist)
    plt.title(f"LR={label}\nTest Acc: {test_acc:.4f}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
plt.tight_layout()
plt.suptitle("Effect of Learning Rate — Loss Curves", y=1.02)
plt.show()

# Confusion matrices for LR
plt.figure(figsize=(5*n, 4))
for i, (label, test_acc, _, _, all_preds, all_labels) in enumerate(lr_data):
    cm = confusion_matrix(all_labels, all_preds)
    plt.subplot(1, n, i+1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10),
                cbar=False, annot_kws={"size": 6})
    plt.title(f"LR={label}\nTest Acc: {test_acc:.4f}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
plt.tight_layout()
plt.suptitle("Effect of Learning Rate — Confusion Matrices", y=1.02)
plt.show()

# EXPERIMENT D: Vary Batch Size
print("\n===== Varying Batch Size =====")
batch_sizes = [32, 128, 256]
batch_results = {}
batch_data = []

for bs in batch_sizes:
    print(f"\nBatch Size: {bs}")
    test_acc, acc_hist, loss_hist, preds, labels = train_and_evaluate(batch_size=bs, epochs=10)
    batch_results[bs] = test_acc
    batch_data.append((str(bs), test_acc, acc_hist, loss_hist, preds, labels))
    print(f"  --> Test Accuracy: {test_acc:.4f}")

plot_experiment("Batch Size", "Batch", batch_sizes, batch_data)

# EXPERIMENT E: Vary Number of Epochs
print("\n===== Varying Number of Epochs =====")
epoch_counts = [5, 10, 20]
epoch_results = {}
epoch_data = []

for ep in epoch_counts:
    print(f"\nEpochs: {ep}")
    test_acc, acc_hist, loss_hist, preds, labels = train_and_evaluate(epochs=ep)
    epoch_results[ep] = test_acc
    epoch_data.append((str(ep), test_acc, acc_hist, loss_hist, preds, labels))
    print(f"  --> Test Accuracy: {test_acc:.4f}")

plot_experiment("Number of Epochs", "Epochs", epoch_counts, epoch_data)

# BEST MODEL: Auto-select from results, then plot confusion matrix
print("\n===== Finding Best Configuration =====")

best_activation = max(act_results,   key=act_results.get)
best_hidden     = max(size_results,  key=size_results.get)
best_lr         = max(lr_results,    key=lr_results.get)
best_batch      = max(batch_results, key=batch_results.get)
best_epochs     = max(epoch_results, key=epoch_results.get)

print(f"  Best Activation  : {best_activation} → {act_results[best_activation]:.4f}")
print(f"  Best Hidden Size : {best_hidden}  → {size_results[best_hidden]:.4f}")
print(f"  Best LR          : {best_lr}   → {lr_results[best_lr]:.4f}")
print(f"  Best Batch Size  : {best_batch}   → {batch_results[best_batch]:.4f}")
print(f"  Best Epochs      : {best_epochs}    → {epoch_results[best_epochs]:.4f}")

print(f"\n===== Training Best Model =====")
print(f"Config: activation={best_activation} | hidden={best_hidden} | "
      f"lr={best_lr} | batch={best_batch} | epochs={best_epochs}")

best_acc, _, _, all_preds, all_labels = train_and_evaluate(
    activation=best_activation,
    hidden_size=best_hidden,
    lr=best_lr,
    batch_size=best_batch,
    epochs=best_epochs
)
print(f"\nBest Model Test Accuracy: {best_acc:.4f}")

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix - Best Model\n"
          f"activation={best_activation} | hidden={best_hidden} | "
          f"lr={best_lr} | batch={best_batch} | epochs={best_epochs}\n"
          f"Test Accuracy: {best_acc:.4f}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# SUMMARY TABLE
print("\n" + "="*45)
print("           SUMMARY OF RESULTS")
print("="*45)

print("\n[A] Activation Functions:")
for k, v in act_results.items():
    marker = " ◄ best" if k == best_activation else ""
    print(f"    {k:8s} : {v:.4f}{marker}")

print("\n[B] Hidden Layer Sizes:")
for k, v in size_results.items():
    marker = " ◄ best" if k == best_hidden else ""
    print(f"    {k:5d}    : {v:.4f}{marker}")

print("\n[C] Learning Rates:")
for k, v in lr_results.items():
    marker = " ◄ best" if k == best_lr else ""
    print(f"    {k:.4f}   : {v:.4f}{marker}")
print("    * LR=0.1 may show unstable training due to large gradient updates")

print("\n[D] Batch Sizes:")
for k, v in batch_results.items():
    marker = " ◄ best" if k == best_batch else ""
    print(f"    {k:5d}    : {v:.4f}{marker}")

print("\n[E] Epoch Counts:")
for k, v in epoch_results.items():
    marker = " ◄ best" if k == best_epochs else ""
    print(f"    {k:4d}     : {v:.4f}{marker}")

print(f"\n[Best Model] activation={best_activation} | hidden={best_hidden} | "
      f"lr={best_lr} | batch={best_batch} | epochs={best_epochs}")
print(f"             Test Accuracy: {best_acc:.4f}")
print("="*45)