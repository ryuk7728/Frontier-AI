import numpy as np

def predict(X,mean=5148.28,SD=2176.98,w=0.2017,b=15.303):
    new_inp = (X-mean)/SD
    y = w*new_inp+b
    return np.expm1(y)


if __name__ == "__main__":
    print(predict(5400))