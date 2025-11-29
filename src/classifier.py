from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

class LeafClassifier:
    def __init__(self, k=5):
        self.scaler = StandardScaler()
        self.knn = KNeighborsClassifier(n_neighbors=k)

    def fit(self, X_train, y_train):
        X_scaled = self.scaler.fit_transform(X_train)
        self.knn.fit(X_scaled, y_train)

    def predict(self, X_test):
        X_scaled = self.scaler.transform(X_test)
        return self.knn.predict(X_scaled)
