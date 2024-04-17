import numpy as np
import matplotlib.pyplot as plt

def plot_train_test(X_train, y_train, X_test, y_test, sample=500):
    # Plot training and testing data
    plt.plot(np.arange(0, 500), X_train[-sample:], label="Training data")
    plt.plot(np.arange(0, 500), y_train[-sample:], label="Training ground truth")
    plt.plot(np.arange(500, 500 + len(X_test)), X_test, label="Testing data")
    plt.plot(np.arange(500, 500 + len(y_test)), y_test, label="Testing ground truth")
    plt.legend()
    plt.show()

def plot_results(y_pred, y_test, sample=500):
    # Plot forecasting results
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(sample), y_pred[:sample], lw=3, label="ESN Prediction")
    plt.plot(np.arange(sample), y_test[:sample], linestyle="--", lw=2, label="True Value")
    plt.plot(np.abs(y_test[:sample] - y_pred[:sample]), label="Absolute deviation")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Forecasting")
    plt.show()

def forecast(reservoir, readout, X_train, Y_train, X_test, Y_test):
    esn = reservoir >> readout

    # Fit ESN and train
    esn.fit(X_train, Y_train)
    results = esn.run(X_test)

    mse = np.mean((Y_test - results) ** 2)
    rmse = np.sqrt(mse)

    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")

    plot_results(results, Y_test, sample=len(Y_test))