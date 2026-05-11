import numpy as np

def raw_pixels(X):
    return X
#epistrefw ta 784 features kathe eikonas dld ta pixels

def row_col_features(X):
    n_samples=X.shape[0]
    images=X.reshape(n_samples,28,28)
    row_mean=images.mean(axis=2) #m.o grammhs
    col_mean=images.mean(axis=1) #m.o sthlhs
    features=np.hstack((row_mean, col_mean))
    return features

def downsample(X):#spaw thn eikona se 2x2 blocks kai pairnw mo kathe block
    n_samples=X.shape[0]
    images=X.reshape(n_samples,28,28)
    downsampled=images.reshape(n_samples,14,2,14,2).mean(axis=(2,4))
    return downsampled.reshape(n_samples, 196)

def combined_features(X):
    #raw=raw_pixels(X)
    raw_c=row_col_features(X)
    down=downsample(X)
    return np.hstack((down, raw_c))