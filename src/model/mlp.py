from src.links.linear import *
from chainer.functions import relu
import numpy as np

class MLP(chainer.Chain):
    # c: 0 for normal training, positive value for Lipschitz Margin Training
    def __init__(self, c=0):
        super(MLP, self).__init__()
        self.c = np.float32(c)
        with self.init_scope():
            self.l0 = Linear(100)
            self.l1 = Linear(100)
            self.l2 = Linear(10)

    def __call__(self, x, use_lmt=True):
        y, l = self.l0((x, self.c * use_lmt))
        y = relu(y)
        y, l = self.l1((y, l))
        y = relu(y)
        return self.l2((y, l))

    def accuracy(self, x, t):
        y, _ = self.__call__(x)
        y = np.argmax(y.data, axis=1)
        return np.sum(y == t) / len(t)
