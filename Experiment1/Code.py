import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix, accuracy_score


class Perceptron:

    def __init__(self, lr=1.0, epochs=6, sleep_time=1.0):
        self.lr = lr
        self.epochs = epochs
        self.sleep_time = sleep_time

    def step(self, z):
        return 1 if z >= 0 else 0

  #Decion Boundry Visualizer
    def plot_boundary(self, ax, X, y, title):

        ax.clear()
        ax.set_title(title)

        # plot points
        for i in range(len(X)):
            if y[i] == 1:
                ax.scatter(X[i][0], X[i][1], marker="o", s=120, label="Class 1" if i==0 else "")
            else:
                ax.scatter(X[i][0], X[i][1], marker="x", s=120, label="Class 0" if i==0 else "")

        # decision boundary  w^T x = 0
        # w0 + w1*x + w2*y = 0  -> y = -(b + w1*x)/w2
        if self.w[2] != 0:
            x_vals = np.array([-1,2])
            y_vals = -(self.w[0] + self.w[1]*x_vals) / self.w[2]
            ax.plot(x_vals, y_vals, 'k--', label="w^T x = 0")

        ax.set_xlim(-0.5,1.5)
        ax.set_ylim(-0.5,1.5)
        ax.legend()

        plt.draw()
        plt.pause(0.01)

  #Train Function

    def fit(self, X, y, title="DATA"):

        X_bias = np.c_[np.ones(X.shape[0]), X]
        self.w = np.zeros(X_bias.shape[1])
        losses = []

        print("\n\n====== TRAINING ON", title, "======")

        plt.ion()
        fig, ax = plt.subplots()

        for epoch in range(self.epochs):

            print("\n====================================")
            print(f"Epoch {epoch+1}")
            print("====================================")

            table_rows = []
            error_count = 0

            for i in range(len(X_bias)):

                x = X_bias[i]
                x1, x2 = X[i]
                y_actual = y[i]

                z = np.dot(self.w, x)
                y_pred = self.step(z)

                error = y_actual - y_pred

                # update rule
                self.w = self.w + self.lr * error * x

                if error != 0:
                    error_count += 1

                # Table Format
                table_rows.append([
                    x1,
                    x2,
                    round(self.w[1],2),
                    round(self.w[2],2),
                    round(self.w[0],2),
                    y_actual,
                    y_pred
                ])

                # animated decision boundry
                self.plot_boundary(ax, X, y, f"{title} - Epoch {epoch+1}")
                time.sleep(self.sleep_time)

            losses.append(error_count)

            df = pd.DataFrame(
                table_rows,
                columns=["x1","x2","w1","w2","b","y_actual","y_pred"]
            )

            print(df.to_string(index=False))

        plt.ioff()
        plt.show()

        return losses
      
      #Predict

    def predict(self, X):
        X_bias = np.c_[np.ones(X.shape[0]), X]
        preds = []
        for x in X_bias:
            preds.append(self.step(np.dot(self.w,x)))
        return np.array(preds)


#Datasets

X_nand = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])
y_nand = np.array([1,1,1,0])

X_xor = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])
y_xor = np.array([0,1,1,0])


#Loss Curve Function

def plot_loss(losses, title):
    plt.figure()
    plt.plot(losses, marker='o')
    plt.title(f"Training Loss Curve - {title}")
    plt.xlabel("Epoch")
    plt.ylabel("Errors")
    plt.show()


if __name__ == "__main__":

    # ================= NAND =================
    model1 = Perceptron(lr=1.0, epochs=6, sleep_time=1.0)
    losses1 = model1.fit(X_nand, y_nand, title="NAND")

    pred_train = model1.predict(X_nand)

    print("\nConfusion Matrix (NAND):")
    print(confusion_matrix(y_nand, pred_train))

    print("Training Accuracy:", accuracy_score(y_nand, pred_train))

    plot_loss(losses1, "NAND")


  
    model2 = Perceptron(lr=1.0, epochs=6, sleep_time=1.0)
    losses2 = model2.fit(X_xor, y_xor, title="XOR")

    pred_train2 = model2.predict(X_xor)

    print("\nConfusion Matrix (XOR):")
    print(confusion_matrix(y_xor, pred_train2))

    print("Training Accuracy:", accuracy_score(y_xor, pred_train2))

    plot_loss(losses2, "XOR")
