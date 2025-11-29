import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def load_features(path):
    df = pd.read_csv(path)
    X = df.drop("class", axis=1).values
    y = df["class"].values
    return X, y

def plot_confusion_matrix(cm, labels, savepath=None):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    plt.show()

def export_csv(df, filename):
    df.to_csv(filename, index=False)
