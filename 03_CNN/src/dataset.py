import numpy as np
import pandas as pd


def one_hot(y):
    return np.eye(2)[y]

def load_dataset(path="data",train_limit=100,test_limit=100):
    train_path = path + r"\mnist_train.csv"
    test_path = path + r"\mnist_test.csv"

    train = pd.read_csv(train_path).values
    test = pd.read_csv(test_path).values

    print(train.shape,test.shape)

    X_train, y_train = preprocess(train,train_limit)
    X_test, y_test = preprocess(test,test_limit)

    X_train = X_train.astype(np.float64)/255
    X_test = X_test.astype(np.float64)/255

    print("Train:",X_train.shape,y_train.shape)
    print("Test:",X_test.shape,y_test.shape)

    return X_train,X_test,y_train,y_test


def preprocess(vals,limit):
    # vals = vals[:]
    mask = vals[:,0]
    mask = (mask == 0) | (mask == 1)
    filtered_vals = vals[mask]

    if limit is None:
        limit = len(filtered_vals)
        print("The new limit is:", limit)

    indices = np.random.choice(filtered_vals.shape[0],limit,replace=False)
    X = filtered_vals[indices]
    y = X[:,0]
    X = X[:,1:]
    y = one_hot(y)

    X = X.reshape(limit,1,28,28)
    y = y.reshape(limit,2,1)



    return X,y



if __name__ == "__main__":
    load_dataset()