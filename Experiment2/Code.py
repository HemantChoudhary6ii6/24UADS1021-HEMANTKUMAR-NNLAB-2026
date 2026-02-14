import numpy as np
import matplotlib.pyplot as plt

class MLP_XOR:

    def __init__(self, input_size=2, hidden_size=4, output_size=1, lr=0.5):

        np.random.seed(1)
        self.lr = lr

        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1/input_size)
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1/hidden_size)
        self.b2 = np.zeros((1, output_size))

        self.loss_history = []
        self.accuracy_history = []

    # Activation 
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # Forward-
    def forward(self, X):
        self.hidden_input = np.dot(X, self.W1) + self.b1
        self.hidden_output = self.sigmoid(self.hidden_input)

        self.final_input = np.dot(self.hidden_output, self.W2) + self.b2
        self.output = self.sigmoid(self.final_input)

        return self.output

    # Backward 
    def backward(self, X, y):

        error = y - self.output

        # Loss
        loss = np.mean(np.square(error))
        self.loss_history.append(loss)

        d_output = error * self.sigmoid_derivative(self.output)

        hidden_error = np.dot(d_output, self.W2.T)
        d_hidden = hidden_error * self.sigmoid_derivative(self.hidden_output)

        # Weight updates
        self.W2 += np.dot(self.hidden_output.T, d_output) * self.lr
        self.b2 += np.sum(d_output, axis=0, keepdims=True) * self.lr

        self.W1 += np.dot(X.T, d_hidden) * self.lr
        self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * self.lr

    # Accuracy 
    def compute_accuracy(self, X, y):
        preds = (self.forward(X) > 0.5).astype(int)
        acc = np.mean(preds == y)
        self.accuracy_history.append(acc)

    # Train 
    def train(self, X, y, epochs=20000):

        for epoch in range(epochs):

            self.forward(X)
            self.backward(X, y)
            self.compute_accuracy(X, y)

            if epoch % 2000 == 0:
                print(f"Epoch {epoch}, Loss: {self.loss_history[-1]:.4f}, Accuracy: {self.accuracy_history[-1]*100:.2f}%")

    # Predict 
    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)



# XOR DATASET

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])


# TRAIN MODEL

model = MLP_XOR(hidden_size=4, lr=0.5)
model.train(X, y, epochs=20000)


# FINAL PREDICTIONS

predictions = model.predict(X)
print("\nPredictions:\n", predictions)


# ACCURACY

accuracy = np.mean(predictions == y) * 100
print("\nFinal Accuracy:", accuracy, "%")


# CONFUSION MATRIX

def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[TN, FP],
                     [FN, TP]])

cm = confusion_matrix(y, predictions)
print("\nConfusion Matrix:\n", cm)


# LOSS CURVE

plt.figure()
plt.plot(model.loss_history)
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()


# EPOCHS vs ACCURACY GRAPH

plt.figure()
plt.plot(model.accuracy_history)
plt.title("Accuracy vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()


# DECISION BOUNDARY VISUALIZATION

def plot_decision_boundary(model, X, y):

    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:,0], X[:,1], c=y.flatten())

    plt.title("Decision Boundary for XOR (MLP)")
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    plt.show()

plot_decision_boundary(model, X, y)
