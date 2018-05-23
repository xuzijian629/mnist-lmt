from src.functions.norm import *
import numpy as np
import chainer

class Linear(chainer.links.Linear):
    def __init__(self, in_size, out_size=None, nobias=False, initialW=None, initial_bias=None):
        super(Linear, self).__init__(in_size, out_size=out_size, nobias=nobias, initialW=initialW, initial_bias=initial_bias)

    def __call__(self, x):
        x, l = x
        x = super(Linear, self).__call__(x)
        if l:
            l = l * L2(self.W)
        return x, l
