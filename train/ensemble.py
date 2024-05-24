import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split


def generate_feature_vectors():
    vectors = []
    for _ in range(100):
        vec = []
        for _ in range(4096):
            x = np.random.rand()
            vec.append(x)
        vectors.append(vec)
    return vectors


def generate_labels():
    labels = []
    for i in range(100):
        labels.append(0 if i % 2 == 0 else 1)

    return labels


def train_classifier(X_train, X_test, y_train, y_test):
    # Convert the dataset into DMatrix, which is a data structure optimized for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Set parameters for the XGBoost model
    params = {
        'objective': 'multi:softmax',
        'max_depth': 6,  # Depth of the trees
        'eta': 0.1,  # Learning rate
    }

    # Train the model
    num_rounds = 100  # Number of boosting rounds
    bst = xgb.train(params, dtrain, num_rounds) # use eval metric here also

    # Predict the labels for the test set
    y_pred = bst.predict(dtest)
    y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions

    # Calculate accuracy
    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')


if __name__ == "__main__":
    feature_vectors = generate_feature_vectors()
    labels = generate_labels()
    X_train, X_test, y_train, y_test = train_test_split(feature_vectors, labels, test_size=0.2, random_state=42)

    train_classifier(X_train, X_test, y_train, y_test)
