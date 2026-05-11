import numpy as np
from sklearn.model_selection import train_test_split
from data_loader import load_mnist
from features import raw_pixels, row_col_features, combined_features, downsample
from knn import KNN_classifier
from nearest_centroid import NearestCentroid
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def show_samples(X, y):
    plt.figure(figsize=(8, 8))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(X[i].reshape(28, 28), cmap='gray')
        plt.title(f"Label: {y[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def show_correct_and_wrong(X, y_true, y_pred):
    correct_idx = None
    wrong_idx = None

    for i in range(len(y_true)):
        if correct_idx is None and y_pred[i] == y_true[i]:
            correct_idx = i
        if wrong_idx is None and y_pred[i] != y_true[i]:
            wrong_idx = i
        if correct_idx is not None and wrong_idx is not None:
            break

    plt.figure(figsize=(6,3))

    # swsth taxinomisi
    plt.subplot(1,2,1)
    plt.imshow(X[correct_idx].reshape(28,28), cmap='gray')
    plt.title(f"Correct\nTrue: {y_true[correct_idx]}, Pred: {y_pred[correct_idx]}")
    plt.axis('off')

    # lathos taxinomisi
    plt.subplot(1,2,2)
    plt.imshow(X[wrong_idx].reshape(28,28), cmap='gray')
    plt.title(f"Wrong\nTrue: {y_true[wrong_idx]}, Pred: {y_pred[wrong_idx]}")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    X, y = load_mnist()

    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)
    print("Unique Labels:", np.unique(y))

    #show_samples(X, y)

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    print("Train labels:", np.bincount(y_train))
    print("Validation labels:", np.bincount(y_val))
    print("Test labels:", np.bincount(y_test))

    # Features
    X_train_raw = raw_pixels(X_train)
    X_val_raw = raw_pixels(X_val)
    X_test_raw = raw_pixels(X_test)

    X_train_rc = row_col_features(X_train)
    X_val_rc = row_col_features(X_val)
    X_test_rc = row_col_features(X_test)

    X_train_down = downsample(X_train)
    X_val_down = downsample(X_val)
    X_test_down = downsample(X_test)

    X_train_comb = combined_features(X_train)
    X_val_comb = combined_features(X_val)
    X_test_comb = combined_features(X_test)

    print("\n--- Feature Shapes ---")
    print("Raw train:", X_train_raw.shape)
    print("Raw val:", X_val_raw.shape)
    print("Raw test:", X_test_raw.shape)

    print("Row/Col train:", X_train_rc.shape)
    print("Row/Col val:", X_val_rc.shape)
    print("Row/Col test:", X_test_rc.shape)

    print("Downsampled train:", X_train_down.shape)
    print("Downsampled val:", X_val_down.shape)
    print("Downsampled test:", X_test_down.shape)

    print("Combined train:", X_train_comb.shape)
    print("Combined val:", X_val_comb.shape)
    print("Combined test:", X_test_comb.shape)

    # Small subsets
    y_train_small = y_train[:1000]
    y_val_small = y_val[:100]

    #sygkrisi gia diaforetika k
    print("\n=== k-NN: Accuracy vs k for all feature sets ===")

    k_values = [1, 3, 5, 7, 9]

    feature_sets = {
        "Raw": (X_train_raw, X_val_raw),
        "Row/Col": (X_train_rc, X_val_rc),
        "Downsampled": (X_train_down, X_val_down),
        "Combined": (X_train_comb, X_val_comb)
    }

    for feature_name, (Xtr, Xval) in feature_sets.items():
        print(f"\n--- {feature_name} Features ---")

        Xtr_small = Xtr[:1000]
        Xval_small = Xval[:100]

        ytr_small = y_train[:1000]
        yval_small = y_val[:100]

        for k in k_values:
            knn = KNN_classifier(k=k)
            knn.fit(Xtr_small, ytr_small)
            y_pred = knn.predict(Xval_small)
            acc = np.mean(y_pred == yval_small)

            print(f"k = {k}, accuracy = {acc}")

    # Final comparison
    print("\n=== FINAL COMPARISON: k-NN vs Nearest Centroid ===")

    feature_sets = {
        "Raw": (X_train_raw, X_val_raw),
        "Row/Col": (X_train_rc, X_val_rc),
        "Downsampled": (X_train_down, X_val_down),
        "Combined": (X_train_comb, X_val_comb)
    }

    for name, (Xtr, Xval) in feature_sets.items():
        print(f"\n--- {name} Features ---")

        Xtr_small = Xtr[:1000]
        Xval_small = Xval[:100]

        # k-NN
        knn = KNN_classifier(k=3) #to epilegw epeidh parolo pou exei to kalytero pososto epityxias mazi me to k=1 sto subset,se oloklhro to dataset tha htan kalytero to k=3
        knn.fit(Xtr_small, y_train_small)
        y_pred_knn = knn.predict(Xval_small)
        acc_knn = np.mean(y_pred_knn == y_val_small)
        print(f"k-NN accuracy: {acc_knn}")

        # Nearest Centroid
        nc = NearestCentroid()
        nc.fit(Xtr_small, y_train_small)
        y_pred_nc = nc.predict(Xval_small)
        acc_nc = np.mean(y_pred_nc == y_val_small)
        print(f"Nearest Centroid accuracy: {acc_nc}")

    #confusion matrix gia euresi lathwn
    #epilegw mono ta kalytera features gia k-NN kai to raw gia Nearest Centroid giati exei to kalytero pososto epityxias se oloklhro to dataset kai sto subset,alla kai giati exei to kalytero pososto epityxias se oloklhro to dataset kai sto subset
    print("\n--- Confusion Matrix: k-NN (Raw) ---")
    knn = KNN_classifier(k=3)
    knn.fit(X_train_raw[:1000], y_train[:1000])
    y_pred = knn.predict(X_val_raw[:200])
    show_correct_and_wrong(X_val_raw[:100], y_val[:100], y_pred)
    cm_knn_raw = confusion_matrix(y_val[:200], y_pred)
    print(cm_knn_raw)

    print("\n--- Confusion Matrix: k-NN (Downsampled) ---")
    knn = KNN_classifier(k=3)
    knn.fit(X_train_down[:1000], y_train[:1000])
    y_pred = knn.predict(X_val_down[:200])
    cm_knn_down = confusion_matrix(y_val[:200], y_pred)
    print(cm_knn_down)

    print("\n--- Confusion Matrix: Nearest Centroid (Raw) ---")
    nc = NearestCentroid()
    nc.fit(X_train_raw[:1000], y_train[:1000])
    y_pred = nc.predict(X_val_raw[:200])
    cm_nc = confusion_matrix(y_val[:200], y_pred)
    print(cm_nc)


if __name__ == "__main__":
    main()