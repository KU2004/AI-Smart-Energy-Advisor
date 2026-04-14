import matplotlib.pyplot as plt
import os

def plot_results(y_test, y_pred):
    os.makedirs("images", exist_ok=True)

    plt.figure(figsize=(10,5))
    plt.plot(y_test.values, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.legend()
    plt.title("Energy Consumption Forecast")

    plt.savefig("images/prediction.png")
    plt.show()

    print("✅ Graph Saved")