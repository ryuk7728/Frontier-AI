import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    
    X = np.array(df["area"]).reshape(-1,1)
    y = np.array(df["price"]).reshape(-1,1)

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = np.log1p(y_train)
    y_test = np.log1p(y_test)

    return X_train,X_test,y_train,y_test

if __name__ == "__main__":
    load_data(r"data\Housing.csv")
