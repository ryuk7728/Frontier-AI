import pandas as pd
import numpy as np

def one_hot(y):
    hot = np.eye(10)[y]
    return hot
    
def load_data(path, noise_samples=0, shuffle=True, seed=42):
    train_path = path + r"\mnist_train.csv"
    test_path = path + r"\mnist_test.csv"
    df_train = pd.read_csv(rf"{train_path}")
    df_test = pd.read_csv(rf"{test_path}")

    X_train = df_train.iloc[:,1:785].values.T / 255
    y_train = df_train.iloc[:,0].values

    X_test = df_test.iloc[:,1:785].values.T / 255
    y_test = df_test.iloc[:,0].values

    y_train = one_hot(y_train).T
    y_test = one_hot(y_test).T

    rng = np.random.default_rng(seed)

    if noise_samples > 0:
        X_noise = rng.uniform(0, 1, (784, noise_samples))
        y_noise = np.ones((10, noise_samples)) / 10

        X_train = np.hstack((X_train, X_noise))
        y_train = np.hstack((y_train, y_noise))

    if shuffle:
        indices = rng.permutation(X_train.shape[1])
        X_train = X_train[:, indices]
        y_train = y_train[:, indices]


    return X_train,X_test,y_train,y_test    

if __name__ == "__main__":
    print(load_data("data", noise_samples=1000)[0].shape)
