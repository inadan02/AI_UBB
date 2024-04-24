from random import random
from math import exp


def sigmoid(x):
    return 1 / (1 + exp(-x))


class MyLogisticRegressionMultipleLabels:

    def __init__(self):
        self.intercept_ = []
        self.coef_ = []

    def fit_batch(self, x, y, learning_rate=0.001, no_epochs=1000):
        self.intercept_ = []
        self.coef_ = []
        labels = list(set(y))
        for label in labels:
            coef = [random() for _ in range(len(x[0]) + 1)]
            for epoch in range(no_epochs):
                #errors = [0] * len(coef)
                errors =[]

                for i in range(len(x)):
                    y_computed = sigmoid(self.evaluate(x[i], coef))
                    if y[i] == label:
                        errors.append(y_computed - 1)
                    else:
                        errors.append(y_computed)
                    #for i, xi in enumerate([1] + list(input)):
                        #errors[i] += error * xi
                    #errors[i]+=error
                #for i in range(len(coef)):
                for i in range(len(x)):
                    for j in range(0, len(x[0])):
                        coef[j] = coef[j] - learning_rate * errors[i]*x[i][j]
                    coef[len(x[0])]=coef[len(x[0])]-learning_rate*errors[len(x[0])]*1
            self.intercept_.append(coef[0])
            self.coef_.append(coef[1:])

    def fit(self, x, y, learning_rate=0.001, no_epochs=1000):
        #self.intercept_ = []
        #self.coef_ = []
        output_labels = list(set(y))
        for label in output_labels:
            coef = [0.0 for _ in range(1 + len(x[0]))]
            #coef = [random() for _ in range(len(x[0]) + 1)]
            for epoch in range(no_epochs):
                for i in range(len(x)):
                    y_computed = sigmoid(self.evaluate(x[i], coef))
                    if y[i] == label:
                        error = y_computed - 1#functia sigmoid duce tot in interval (0,1), 2 variante->e label bun sau nu
                    else:
                        error = y_computed
                    #error=y_computed-(y[i]==label)
                    for j in range(0,len(x[0])):
                        coef[j + 1] = coef[j + 1] - learning_rate * error * x[i][j]
                    coef[0] = coef[0] - learning_rate * error
            self.intercept_.append(coef[0])
            self.coef_.append(coef[1:])

    def evaluate(self, xi, coefficient):
        yi = coefficient[0]
        for j in range(len(xi)):
            yi += coefficient[j + 1] * xi[j]
        return yi

    def predict_one_sample(self, sample_features):
        #A high threshold value will result in high precision but low recall, while a low threshold value will result in high recall but low precision.
        #threshold = 0.5 binar->0 sau 1
        computed_labels = []
        threshold = 0.5
        for intercept, coefficient in zip(self.intercept_, self.coef_):
            computed_float_value = self.evaluate(sample_features, [intercept] + coefficient)
            computed_01_value=sigmoid(computed_float_value)
            computed_labels.append(computed_01_value)
        return computed_labels.index(max(computed_labels))



    def predict(self, in_test):
        computed_labels = [self.predict_one_sample(sample) for sample in in_test]
        return computed_labels
