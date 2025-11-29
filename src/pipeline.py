from utils import load_features, plot_confusion_matrix
from classifier import LeafClassifier
from sklearn.metrics import confusion_matrix, classification_report

def main():

    X_train, y_train = load_features("data/train_features.csv")
    X_test, y_test   = load_features("data/test_features.csv")

    clf = LeafClassifier(k=5)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)

    print("\n=== RELATÓRIO DE CLASSIFICAÇÃO ===\n")
    print(classification_report(y_test, preds))

    cm = confusion_matrix(y_test, preds)
    plot_confusion_matrix(cm,
                          labels=["Black Rot","ESCA","Healthy","Leaf Blight"],
                          savepath="results/confusion_matrix.png")

if __name__ == "__main__":
    main()
