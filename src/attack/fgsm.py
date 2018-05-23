import numpy as np
import chainer

class FGSM:
    def __init__(self, model):
        self.model = model

    def confidence(self, x, t):
        c, _ = self.model(np.array([x]), use_lmt=False)
        return c.data[0, t]

    def is_correct(self, x, t):
        c, _ = self.model(np.array([x]), use_lmt=False)
        return np.argmax(c.data[0]) == t

    def attack(self, x, t, eta=0.05):
        if not self.is_correct(x, t):
            return -1,
        conf = self.confidence(x, t)
        grads = []
        for i in range(len(x)):
            x[i] += eta
            grads.append(np.sign(self.confidence(x, t) - conf))
            x[i] -= eta
        grads = np.array(grads)
        adv = np.clip(x - eta * grads, 0, 1 - 1e-6)
        return self.is_correct(adv, t)
