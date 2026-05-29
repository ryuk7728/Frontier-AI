class LinearRegression:

    def __init__(self):
        self.w = 0
        self.b = 0

    def predict(self,X):
        return self.w*X + self.b
    