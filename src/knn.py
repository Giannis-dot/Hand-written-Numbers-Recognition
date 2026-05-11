import numpy as np
from collections import Counter

class KNN_classifier:
    def __init__(self, k=20):
        self.k=k
        self.X_train=None
        self.y_train=None


    def fit(self,X,y):
        self.X_train=X
        self.y_train=y

    def euclidean_distance(self,x1,x2):
        return np.sqrt(np.sum((x1-x2)**2))

    def predict_one(self, x):
        distances = []
        for i in range(len(self.X_train)):
            dist = self.euclidean_distance(x, self.X_train[i])
            distances.append((dist, self.y_train[i]))
        distances.sort(key=lambda pair: pair[0])
        k_nearest_labels = [label for _, label in distances[:self.k]]
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            pred = self.predict_one(x)
            predictions.append(pred)
        return np.array(predictions)