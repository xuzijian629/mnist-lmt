import numpy as np
import chainer

class Lmt(chainer.function.Function):
    def __init__(self, t):
        self.t = t

    def forward(self, x):
        x, l = x
        assert l.shape == (1,)
        l = l * np.sqrt(2)
        t = self.t
        x2 = x.copy()
        x2[list(range(t.size)), t] = np.min(x, 1)
        self.factor = np.clip((x[list(range(t.size)), t] - np.max(x2, 1)) / l, 0, 1)
        x2[list(range(t.size)), t] = x[list(range(t.size)), t] - self.factor * l
        return x2, l

    def backward(self, inputs, grad_outputs):
        gy, gl = grad_outputs
        assert gl is None
        return gy, -(gy[list(range(self.t.size)), self.t] * self.factor).sum().reshape((1,)) * np.sqrt(2)
