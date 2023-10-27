import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

def plot_histogram(data, threshold=None, bins=10, x_label='Values', y_label='Frequency'):

    plt.figure(figsize=(8, 6))
    plt.rcParams["font.family"] = "serif"

    if threshold:
        plt.axvline(threshold, color='red', linestyle='--', label='Upper Bound')

    plt.hist(data, bins=bins)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def plot_overlap_histograms(arrays, labels, column, alpha=0.5, bins=20, xlabel="Score"):
    # Calculate histograms for both arrays
    _, bin_edges = np.histogram(arrays[0][column], bins=bins, density=True)

    # Calculate the bin width
    bin_width = bin_edges[1] - bin_edges[0]
    
    plt.figure(figsize=(8, 6))
    plt.rcParams["font.family"] = "serif"
    
    for array, label in zip(arrays, labels):
        hist, _ = np.histogram(array[column], bins=bin_edges, density=True)

        # Normalize histograms to represent ratios (relative frequencies)
        # hist /= np.sum(hist * bin_width)

        # Plot the histograms
        plt.bar(bin_edges[:-1], hist, width=bin_width, alpha=alpha, label=label)

    plt.xlabel(xlabel)
    plt.ylabel('Relative Frequency')
    plt.legend()
    plt.show()

def plot_boxplot(arrays, labels, column, ylabel):
    plt.figure(figsize=(8, 6))
    plt.rcParams["font.family"] = "serif"

    # Initialize a list to store the positions of the boxplots
    positions = []

    for score, label in zip(arrays, labels):
        # Generate positions for each boxplot
        position = len(positions) + 0.2
        positions.append(position)
        
        # Plot the boxplot at the calculated position
        plt.boxplot(score[column], positions=[position], labels=[label], patch_artist=True, showfliers=False, widths=0.4)

    # Set the x-axis labels
    plt.xticks(positions, labels)

    # Add labels and legend if needed
    plt.ylabel(ylabel)

    plt.show()
    
def plot_roc_curve(models):
    
    plt.figure(figsize=(8, 6))
    plt.rcParams["font.family"] = "serif"

    for model in models:
        name = model["name"]
        fpr = model["fpr"]
        tpr = model["tpr"]
        auc = model["auc"]
        emph = model["emph"]

        if emph:
            plt.plot(
                fpr,
                tpr,
                lw=2,
                color='k',
                label=f'{name} (area = {auc:.2f})'
            )
        else:
            plt.plot(
                fpr,
                tpr,
                lw=2,
                alpha=0.7,
                label=f'{name} (area = {auc:.2f})'
            )

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()

    return

def performance_metrics(targets, predictions, threshold):
    accuracy = accuracy_score(targets, threshold < predictions)
    precision = precision_score(targets, threshold < predictions)
    recall = recall_score(targets, threshold < predictions)
    f1 = f1_score(targets, threshold < predictions)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")

def plot_confusion_matrix(targets, predictions, threshold):

    cm = confusion_matrix(targets, threshold < predictions)
    accuracy = accuracy_score(targets, threshold < predictions)

    plt.figure(figsize=(8, 6))
    plt.rcParams["font.family"] = "serif"
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Accuracy: {accuracy:.2f}")

    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')

    plt.xticks([0, 1], ["Negative", "Positive"])
    plt.yticks([0, 1], ["Negative", "Positive"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), horizontalalignment='center', verticalalignment='center', color='black', fontsize=16)

    plt.show()
