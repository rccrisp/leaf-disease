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

def plot_overlap_histograms(array1, array2, labels, alpha=0.5, bins=20):
    # Calculate histograms for both arrays
    hist1, bin_edges = np.histogram(array1, bins=bins, density=True)
    hist2, _ = np.histogram(array2, bins=bin_edges, density=True)

    # Calculate the bin width
    bin_width = bin_edges[1] - bin_edges[0]

    # Normalize histograms to represent ratios (relative frequencies)
    hist1 /= np.sum(hist1 * bin_width)
    hist2 /= np.sum(hist2 * bin_width)

    # Plot the histograms
    plt.figure(figsize=(8, 6))
    plt.rcParams["font.family"] = "serif"
    plt.bar(bin_edges[:-1], hist1, width=bin_width, alpha=alpha, label=labels[0])
    plt.bar(bin_edges[:-1], hist2, width=bin_width, alpha=alpha, label=labels[1])

    plt.xlabel('Score')
    plt.ylabel('Relative Frequency')
    plt.legend()
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
