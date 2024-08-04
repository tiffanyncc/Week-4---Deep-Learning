import matplotlib.pyplot as plt
import pandas as pd

def plot_loss_curves(history, filename):
    pd.DataFrame(history.history).plot()
    plt.title(f"Model training curves ({filename})")
    plt.savefig(f'src/visualization/images/{filename}.png')
    plt.show()

def plot_learning_rate_vs_loss(lrs, history, filename):
    plt.figure(figsize=(10, 7))
    plt.semilogx(lrs, history.history["loss"]) # we want the x-axis (learning rate) to be log scale
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("Learning rate vs. loss");
    plt.savefig(f'src/visualization/images/{filename}.png')
    plt.show()
