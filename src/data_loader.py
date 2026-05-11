from sklearn.datasets import fetch_openml
import numpy as np

def load_mnist():
    #Loading the Mnist dataset
    X,y = fetch_openml('mnist_784', version=1,return_X_y=True, as_frame=False)
    X = X.astype(np.float32)
    y=y.astype(np.int64)
    X/=255.0
    return X,y
