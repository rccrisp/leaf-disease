import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_train_val(train_epoch, train_metric, val_epoch, val_metric, val_score=None):

    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
    plt.rcParams["font.family"] = "serif"
    sns.set(style="whitegrid")  # Use Seaborn's white grid style

    plt.plot(train_epoch, train_metric, label='Training Loss', color='blue', linestyle='-')
    plt.plot(val_epoch, val_metric, label='Validation Loss', color='red', linestyle='-')
    if val_score:
        plt.plot(val_epoch, val_score, label='Validation Anomaly Score', color='k', linestyle='-')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()  # Display the legend

    plt.show()

def plot_train_val_GAN(train_epoch, train_gen_metric, train_disc_metric, val_epoch, val_gen_metric, val_disc_metric):

    plt.figure(figsize=(8, 6))
    plt.rcParams["font.family"] = "serif"
    sns.set(style="whitegrid")

    fig, ax1 = plt.subplots()

    # Plot generator training and validation loss on the primary y-axis
    ax1.plot(train_epoch, train_gen_metric, label='Generator Training Loss', color='blue', linestyle='-')
    ax1.plot(val_epoch, val_gen_metric, label='Generator Validation Loss', color='blue', linestyle='--')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Generator Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a secondary y-axis for the discriminator
    ax2 = ax1.twinx()
    ax2.plot(train_epoch, train_disc_metric, label='Discriminator Training Loss', color='red', linestyle='-')
    ax2.plot(val_epoch, val_disc_metric, label='Discriminator Validation Loss', color='red', linestyle='--')
    ax2.set_ylabel('Discriminator Loss', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title('GAN Training and Validation Loss')
    fig.tight_layout()
    plt.legend(loc='upper right')

    plt.show()


def plot_activation_functions():
    # Generate a range of values for the x-axis
    x = np.linspace(-5, 5, 100)

    # Calculate the sigmoid and tanh functions
    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)

    # Create a plot
    plt.figure(figsize=(10, 5))
    plt.rcParams["font.family"] = "serif"
    plt.plot(x, sigmoid, label='Sigmoid', linewidth=2)
    plt.plot(x, tanh, label='Tanh', linewidth=2)

    # Add labels and a legend
    plt.title('')
    plt.legend()

    # Remove axis ticks and labels
    plt.xticks([])
    plt.yticks([])
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    # Remove the plot border
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    # Show the plot
    plt.axhline(0, color='black', linewidth=0.5, linestyle='-')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='-')
    plt.show()

