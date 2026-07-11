from .convolutional import Convolutional
from .activations import ReLU,Softmax
from .dense import Dense
from .flatten import Flatten


def build_mnist_cnn():

    return [Convolutional((1,28,28),3,5),
            ReLU(),
            Flatten((5,26,26)),
            Dense(5*26*26,100),
            ReLU(),
            Dense(100,2),
            Softmax()]