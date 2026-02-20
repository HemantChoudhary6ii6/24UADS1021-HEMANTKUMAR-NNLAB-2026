import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load Dataset 

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    as_supervised=True,
    with_info=True
)

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [-1])
    label = tf.one_hot(label, depth=10)
    return image, label

batch_size = 128
ds_train = ds_train.map(preprocess).shuffle(60000).batch(batch_size)
ds_test = ds_test.map(preprocess).batch(batch_size)

#Network Parameters

n_input = 784
n_hidden1 = 256
n_hidden2 = 128
n_output = 10

learning_rate = 0.01
epochs = 20

# He Initialization (Manual)

def he_init(shape):
    stddev = tf.sqrt(2.0 / shape[0])
    return tf.random.normal(shape, stddev=stddev)

W1 = tf.Variable(he_init([n_input, n_hidden1]))
b1 = tf.Variable(tf.zeros([n_hidden1]))

W2 = tf.Variable(he_init([n_hidden1, n_hidden2]))
b2 = tf.Variable(tf.zeros([n_hidden2]))

W3 = tf.Variable(he_init([n_hidden2, n_output]))
b3 = tf.Variable(tf.zeros([n_output]))

# Feed Forward

def forward_pass(X):
    Z1 = tf.matmul(X, W1) + b1
    A1 = tf.nn.relu(Z1)

    Z2 = tf.matmul(A1, W2) + b2
    A2 = tf.nn.relu(Z2)

    Z3 = tf.matmul(A2, W3) + b3
    return Z3

# Loss Function

def compute_loss(logits, labels):
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)
    )

# Training (Backpropagation)

train_loss_history = []     
train_acc_history = []

steps_per_epoch = 60000 // batch_size

for epoch in range(epochs):

    epoch_loss = 0.0
    correct = 0
    total = 0

    iterator = iter(ds_train)

    for step in range(steps_per_epoch):
        X_batch, y_batch = next(iterator)

        with tf.GradientTape() as tape:
            logits = forward_pass(X_batch)
            loss = compute_loss(logits, y_batch)

        gradients = tape.gradient(loss, [W1, b1, W2, b2, W3, b3])

        W1.assign_sub(learning_rate * gradients[0])
        b1.assign_sub(learning_rate * gradients[1])
        W2.assign_sub(learning_rate * gradients[2])
        b2.assign_sub(learning_rate * gradients[3])
        W3.assign_sub(learning_rate * gradients[4])
        b3.assign_sub(learning_rate * gradients[5])

        epoch_loss += loss.numpy()

        preds = tf.argmax(logits, axis=1)
        labels = tf.argmax(y_batch, axis=1)

        correct += tf.reduce_sum(
            tf.cast(preds == labels, tf.float32)
        ).numpy()

        total += X_batch.shape[0]

    avg_loss = epoch_loss / steps_per_epoch

    epoch_accuracy = correct / total
    train_loss_history.append(epoch_loss)

    train_acc_history.append(epoch_accuracy)

    print(f"Epoch {epoch+1}/{epochs} | "
          f"Loss: {avg_loss:.4f} | "
          f"Accuracy: {epoch_accuracy:.4f}")


# Test Accuracy

correct = 0
total = 0
all_preds = []
all_labels = []

for X_batch, y_batch in ds_test:
    logits = forward_pass(X_batch)
    preds = tf.argmax(logits, axis=1)
    labels = tf.argmax(y_batch, axis=1)

    all_preds.extend(preds.numpy())
    all_labels.extend(labels.numpy())

    correct += tf.reduce_sum(
        tf.cast(preds == labels, tf.float32)
    ).numpy()

    total += X_batch.shape[0]

test_accuracy = correct / total
print("\nFinal Test Accuracy:", test_accuracy)

# Plot Loss Curve

plt.figure()
plt.plot(train_loss_history)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Plot Accuracy Curve

plt.figure()
plt.plot(train_acc_history)
plt.title("Training Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()


# Confusion Matrix

cm = confusion_matrix(all_labels, all_preds)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Sample Predictions

plt.figure(figsize=(10,5))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(tf.reshape(list(ds_test)[0][0][i], (28,28)), cmap="gray")
    plt.title(f"Pred: {all_preds[i]}")
    plt.axis("off")
plt.show()
