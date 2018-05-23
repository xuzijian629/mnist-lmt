from src.model.mlp import *
from src.attack.fgsm import *
from src.functions.lmt import *
from chainer.iterators import SerialIterator
from chainer.optimizers import Adam as Opt
from chainer.functions import mean_squared_error as mse

def format(batch):
    xs = np.array([*map(lambda x: x[0], batch)], dtype=np.float32)
    ts = np.array([*map(lambda x: x[1], batch)])
    return xs, ts

def main():
    train, test = chainer.datasets.get_mnist()
    def forward(x, t, model):
        y, l = model(x)
        if model.c:
            y, l = Lmt(t)(y, l)
        t = np.eye(10)[t].astype(np.float32)
        loss = mse(y, t)
        return loss

    model = MLP(c=0.05)
    optimizer = Opt()
    optimizer.setup(model)

    for epoch in range(5):
        for batch in SerialIterator(train, 60, repeat=False):
            x, t = format(batch)
            optimizer.update(forward, x, t, model)
        tx, tt = format(test)
        print("epoch {}: accuracy: {:.3f}".format(epoch + 1, model.accuracy(tx, tt)))

    fgsm = FGSM(model)
    for eta in [0.01, 0.02, 0.05, 0.1]:
        cnt = 0
        fail = 0
        for i in np.random.randint(0, 10000, 100):
            res = fgsm.attack(test[i][0], test[i][1], eta=eta)
            if res != -1:
                cnt += 1
                if not res: fail += 1
        print("c: {:.3f}, eta: {:.3f}, attacked: {:.3f}".format(model.c, eta, fail / cnt))

if __name__ == '__main__':
    main()
