import numpy as np


def train_test_split(X, y, seed, test_size=0.3):
    np.random.seed(seed)

    # Initial set up/helpers
    m_x, m_y = X.shape[0], y.shape[0]
    n_x, n_y = X.shape[1], y.shape[1]

    # 1) Join features and labels and shuffle their order
    dataset = np.column_stack([X, y])
    np.random.shuffle(dataset)

    # 2) Split the train/test sets
    n = round(m_x * test_size)
    test_set = dataset[:n]
    train_set = dataset[n:]

    # 3) Pull the train/test features and labels
    X_train = train_set[:, :n_x]
    y_train = train_set[:, -n_y:]
    X_test = test_set[:, :n_x]
    y_test = test_set[:, -n_y:]
    return (X_train, y_train), (X_test, y_test)
