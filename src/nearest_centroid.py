import numpy as np


class NearestCentroid:
    def __init__(self):
        self.centroids = {} #dictionary me ta noumera klasewn ws keys kai ta onomata centroids ws values

    def fit(self, X_train, y_train):
        classes = np.unique(y_train)

        for c in classes:
            X_c = X_train[y_train == c]
            centroid = np.mean(X_c, axis=0)
            self.centroids[c] = centroid

    def predict_one(self, x):
        distances = {}

        for c, centroid in self.centroids.items():
            dist = np.linalg.norm(x - centroid)
            distances[c] = dist

        return min(distances, key=distances.get)

    def predict(self, X_test):
        predictions = []

        for x in X_test:
            pred = self.predict_one(x)
            predictions.append(pred)

        return np.array(predictions)