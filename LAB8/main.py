from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from sklearn.metrics import accuracy_score
from LogisticRegressionMultipleLabels import MyLogisticRegressionMultipleLabels
from sklearn.linear_model import SGDClassifier

def load_data_flowers():
    data = load_iris()
    inputs = data['data']
    outputs = data['target']
    outputs_name = data['target_names']
    feature_names = list(data['feature_names'])
    feature1 = [feat[feature_names.index('sepal length (cm)')] for feat in inputs]
    feature2 = [feat[feature_names.index('sepal width (cm)')] for feat in inputs]
    feature3 = [feat[feature_names.index('petal length (cm)')] for feat in inputs]
    feature4 = [feat[feature_names.index('petal width (cm)')] for feat in inputs]
    inputs = [[feat[feature_names.index('sepal length (cm)')],
               feat[feature_names.index('sepal width (cm)')],
               feat[feature_names.index('petal length (cm)')],
               feat[feature_names.index('petal width (cm)')]] for feat in inputs]
    return inputs, outputs, outputs_name, feature1, feature2, feature3, feature4, feature_names


def plot_data(inputs, outputs, output_names, feature_names, title=None):
    labels = set(outputs)
    no_data = len(inputs)
    for crt_label in labels:
        x = [inputs[i][0] for i in range(no_data) if outputs[i] == crt_label]
        y = [inputs[i][1] for i in range(no_data) if outputs[i] == crt_label]
        plt.scatter(x, y, label=output_names[crt_label])
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.legend()
    plt.title(title)
    plt.show()


def plot_histogram_feature(feature, variableName):
    plt.hist(feature, 10)
    plt.title('Histogram of ' + variableName)
    plt.show()


def train_and_test(inputs, outputs):
    indexes = [i for i in range(len(inputs))]
    train_sample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    test_sample = [i for i in indexes if i not in train_sample]
    train_inputs = [inputs[i] for i in train_sample]
    train_outputs = [outputs[i] for i in train_sample]
    test_inputs = [inputs[i] for i in test_sample]
    test_outputs = [outputs[i] for i in test_sample]
    return train_inputs, train_outputs, test_inputs, test_outputs


def normalisation(trainData, testData):
    # standardisation/z-normalization
    scaler = StandardScaler()
    if not isinstance(trainData[0], list):
        trainData = [[d] for d in trainData]
        testData = [[d] for d in testData]
        scaler.fit(trainData)
        normalisedTrainData = scaler.transform(trainData)
        normalisedTestData = scaler.transform(testData)
        normalisedTrainData = [el[0] for el in normalisedTrainData]
        normalisedTestData = [el[0] for el in normalisedTestData]
    else:
        scaler.fit(trainData)
        normalisedTrainData = scaler.transform(trainData)
        normalisedTestData = scaler.transform(testData)
    return normalisedTrainData, normalisedTestData


def plot_predictions(inputs, real_outputs, computed_outputs, label_names, feature_names, title):
    labels = list(set(outputs))
    no_data = len(inputs)
    for crt_label in labels:
        x = [inputs[i][0] for i in range(no_data) if real_outputs[i] == crt_label and computed_outputs[i] == crt_label]
        y = [inputs[i][1] for i in range(no_data) if real_outputs[i] == crt_label and computed_outputs[i] == crt_label]
        plt.scatter(x, y, label=label_names[crt_label] + ' (correct)')
    for crt_label in labels:
        x = [inputs[i][0] for i in range(no_data) if real_outputs[i] == crt_label and computed_outputs[i] != crt_label]
        y = [inputs[i][1] for i in range(no_data) if real_outputs[i] == crt_label and computed_outputs[i] != crt_label]
        plt.scatter(x, y, label=label_names[crt_label] + ' (incorrect)')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title(title)
    plt.legend()
    plt.show()


def calculate_performance(computed_outputs, test_outputs):
    error = 0.0
    for t1, t2 in zip(computed_outputs, test_outputs):
        if t1 != t2:
            error += 1
    error = error / len(test_outputs)
    print('Classification error by me: ', error)



def learn_by_tool_multi_label(train_inputs, train_outputs, test_inputs, test_outputs):
    classifier = linear_model.LogisticRegression()
    classifier.fit(train_inputs, train_outputs)
    w0, w1, w2, w3, w4 = classifier.intercept_[0], classifier.coef_[0][0], classifier.coef_[0][1], classifier.coef_[0][
        2], classifier.coef_[0][3]
    print('Classification model by tool Setosa: y =', w0, '+', w1, '* feat1 +', w2, '* feat2 +', w3, '* feat3 +',
          w4, '* feat4')
    w0, w1, w2, w3, w4 = classifier.intercept_[1], classifier.coef_[1][0], classifier.coef_[1][1], classifier.coef_[1][
        2], classifier.coef_[1][3]
    print('Classification model by tool Versicolour: y =', w0, '+', w1, '* feat1 +', w2, '* feat2 +', w3, '* feat3 +',
          w4, '* feat4')
    w0, w1, w2, w3, w4 = classifier.intercept_[2], classifier.coef_[2][0], classifier.coef_[2][1], classifier.coef_[2][
        2], classifier.coef_[2][3]
    print('Classification model by tool Virginica: y =', w0, '+', w1, '* feat1 +', w2, '* feat2 +', w3, '* feat3 +',
          w4, '* feat4')
    computed_outputs = classifier.predict(test_inputs)
    print("Accuracy score:", classifier.score(test_inputs, test_outputs))
    return computed_outputs


def learn_by_me_multi_label(train_inputs, train_outputs, test_inputs, test_outputs):
    classifier = MyLogisticRegressionMultipleLabels()
    # classifier.fit_batch(train_inputs, train_outputs)
    classifier.fit(train_inputs, train_outputs)
    w0, w1, w2, w3, w4 = classifier.intercept_[0], classifier.coef_[0][0], classifier.coef_[
        0][1], classifier.coef_[0][2], classifier.coef_[0][3]
    print('Classification model by me Setosa: y =', w0, '+', w1, '* feat1 +', w2, '* feat2 +', w3, '* feat3 +',
          w4, '* feat4')
    w0, w1, w2, w3, w4 = classifier.intercept_[1], classifier.coef_[1][0], classifier.coef_[
        1][1], classifier.coef_[1][2], classifier.coef_[1][3]
    print('Classification model by me Versicolour: y =', w0, '+', w1, '* feat1 +', w2, '* feat2 +', w3, '* feat3 +',
          w4, '* feat4')
    w0, w1, w2, w3, w4 = classifier.intercept_[2], classifier.coef_[2][0], classifier.coef_[
        2][1], classifier.coef_[2][2], classifier.coef_[2][3]
    print('Classification model by me Virginica: y =', w0, '+', w1, '* feat1 +', w2, '* feat2 +', w3, '* feat3 +',
          w4, '* feat4')
    computed_outputs = classifier.predict(test_inputs)
    no_data = len(test_inputs)
    accuracy = 0.0
    for i in range(no_data):
        if test_outputs[i] == computed_outputs[i]:
            accuracy += 1
    print("Accuracy score:", accuracy / no_data)
    return computed_outputs


def other_loss_function(train_inputs, train_outputs, test_inputs, test_outputs):
    classifier = SGDClassifier(loss='log_loss') # cross entropy
    classifier.fit(train_inputs, train_outputs)
    print('Log loss by tool:', classifier.score(test_inputs, test_outputs))
    classifier = SGDClassifier(loss='huber')  # combination between mean squared error and absolute value function
    classifier.fit(train_inputs, train_outputs)
    print('Huber loss by tool:', classifier.score(test_inputs, test_outputs))
    classifier = SGDClassifier(loss='squared_error')  # MSE
    classifier.fit(train_inputs, train_outputs)
    print('Squared error loss by tool:', classifier.score(test_inputs, test_outputs))


if __name__ == '__main__':
    inputs, outputs, outputNames, feature1, feature2, feature3, feature4, featureNames = load_data_flowers()
    trainInputs, trainOutputs, testInputs, testOutputs = train_and_test(inputs, outputs)
    trainInputs, testInputs = normalisation(trainInputs, testInputs)
    computedTestOutputs = learn_by_tool_multi_label(trainInputs, trainOutputs, testInputs, testOutputs)
    plot_predictions(testInputs, testOutputs, computedTestOutputs, outputNames, featureNames[:2], "Results by tool")
    plot_predictions(testInputs, testOutputs, computedTestOutputs, outputNames, featureNames[2:], "Results by tool")
    print("Classification error by tool: ", 1-accuracy_score(testOutputs, computedTestOutputs))
    print()
    computedTestOutputs = learn_by_me_multi_label(trainInputs, trainOutputs, testInputs, testOutputs)
    plot_predictions(testInputs, testOutputs, computedTestOutputs, outputNames, featureNames[:2], "Results by me")
    plot_predictions(testInputs, testOutputs, computedTestOutputs, outputNames, featureNames[:2], "Results by me")
    calculate_performance(computedTestOutputs, testOutputs)
    print()
    print("Other loss functions:")
    trainInputs, trainOutputs, testInputs, testOutputs = train_and_test(inputs, outputs)
    trainInputs, testInputs = normalisation(trainInputs, testInputs)
    other_loss_function(trainInputs, trainOutputs, testInputs, testOutputs)
