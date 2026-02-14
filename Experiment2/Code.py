import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Generate Large XOR Dataset (10,000 samples)
np.random.seed(1)

N = 10000
X = np.random.rand(N, 2)

y = np.logical_xor(X[:, 0] > 0.5, X[:, 1] > 0.5)
y = y.astype(int).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


class MLP:

    def __init__(self, input_size=2, hidden_size=20, output_size=1, lr=0.3):

        self.lr = lr

        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

        self.loss_history = []
        self.train_accuracy_history = []

    # Sigmoid Activation
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2

    def backward(self, X, y):

        m = y.shape[0]

        # MSE Loss
        error = y - self.a2
        loss = np.mean(np.square(error))
        self.loss_history.append(loss)

        # Output layer gradient (MSE + sigmoid)
        d_output = error * self.sigmoid_derivative(self.a2)

        dW2 = np.dot(self.a1.T, d_output) / m
        db2 = np.sum(d_output, axis=0, keepdims=True) / m

        # Hidden layer gradient
        hidden_error = np.dot(d_output, self.W2.T)
        d_hidden = hidden_error * self.sigmoid_derivative(self.a1)

        dW1 = np.dot(X.T, d_hidden) / m
        db1 = np.sum(d_hidden, axis=0, keepdims=True) / m

        # Weight Updates
        self.W2 += self.lr * dW2
        self.b2 += self.lr * db2
        self.W1 += self.lr * dW1
        self.b1 += self.lr * db1

    def train(self, X, y, epochs=12000):

        for epoch in range(epochs):

            self.forward(X)
            self.backward(X, y)

            preds = (self.a2 > 0.5).astype(int)
            acc = np.mean(preds == y)
            self.train_accuracy_history.append(acc)

            if epoch % 2000 == 0:
                print(f"Epoch {epoch}, Loss: {self.loss_history[-1]:.4f}, Accuracy: {acc*100:.2f}%")

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)

# Train Model

model = MLP(hidden_size=20, lr=0.3)
model.train(X_train, y_train, epochs=12000)

# Evaluate Performance

train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

train_acc = np.mean(train_preds == y_train) * 100
test_acc = np.mean(test_preds == y_test) * 100

print("\nTraining Accuracy:", train_acc, "%")
print("Test Accuracy:", test_acc, "%")

print("\nConfusion Matrix (Test Data):")
print(confusion_matrix(y_test, test_preds))

# Loss Curve

plt.figure()
plt.plot(model.loss_history)
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# Accuracy Curve

plt.figure()
plt.plot(model.train_accuracy_history)
plt.title("Accuracy vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

# Sigmoid Activation Function Graph
def plot_sigmoid():

    x = np.linspace(-10, 10, 400)
    y = 1 / (1 + np.exp(-x))

    plt.figure()
    plt.plot(x, y)
    plt.title("Sigmoid Activation Function")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.grid(True)
    plt.show()

plot_sigmoid()

# Decision Boundary Visualization

def plot_decision_boundary(model, X, y):

    xx, yy = np.meshgrid(
        np.linspace(0, 1, 300),
        np.linspace(0, 1, 300)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), s=10)
    plt.title("Decision Boundary (XOR - 10,000 Samples)")
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    plt.show()

plot_decision_boundary(model, X_test, y_test)
