import numpy as np
from sklearn import linear_model


def tool_univariat(train_inputs, train_outputs):
    xx = []
    for i in train_inputs:
        xx.append([i])

    regressor = linear_model.SGDRegressor()
    regressor.fit(xx, train_outputs)

    w0, w1 = regressor.intercept_[0], regressor.coef_[0]

    print('Regressor calculated by tool: f(x) = ', w0, ' + ', w1, ' * x')
    return w0, w1


class MySGDRegression:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = []

    # simple batch GD
    def fit(self, x, y, learningRate=0.001, noEpochs=1000):
        self.coef_ = [0.0 for _ in range(len(x[0]) + 1)]  # coeficientii w0,w1,.. y = w0 + w1 * x1 + w2 * x2 + ...
        for epoch in range(noEpochs):  # iteram peste epoci
            # amestecam datele ca sa evitam ciclurile
            indexes = [i for i in range(len(x))]  # lista de indici
            indexes_shuffled = np.random.choice(indexes, len(x), replace=False)  # amestecam indicii
            x_shuffled = []
            y_shuffled = []
            for i in indexes_shuffled:
                x_shuffled.append(x[i])  # amestecam datele
                y_shuffled.append(y[i])
            x = x_shuffled
            y = y_shuffled

            crtError = []
            for i in range(len(x)):  # pentru fiecare valoare de training
                ycomputed = self.eval(x[i])  # estimam outputul folosind coeficientii curenti
                crtError.append(ycomputed - y[i])  # calculam eroarea pentru valoarea curenta= prezis-real
            for i in range(len(x)):
                for j in range(0, len(x[0])):  # actualizam coeficientii
                    self.coef_[j] = self.coef_[j] - learningRate * crtError[i] * x[i][j]  # w=w-learningRate*err*x
                # update bias term/intercept term w=w-learningRate*err*1
                self.coef_[len(x[0])] = self.coef_[len(x[0])] - learningRate * crtError[len(x[0])] * 1

        self.intercept_ = self.coef_[-1]  # ultimul coeficient este bias term
        self.coef_ = self.coef_[:-1]  # scoatem bias de la final pentru a avea doar coeficientii

    def eval(self, xi):
        # calculam yi
        yi = self.coef_[-1]  # bias term
        for j in range(len(xi)):
            yi += self.coef_[j] * xi[j]  # w1*x1+w2*x2+...
        return yi

    def predict(self, x):
        yComputed = [self.eval(xi) for xi in x]
        return yComputed
