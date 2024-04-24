import itertools
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from sklearn import neural_network
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

import ANN


def load_data_flowers():
    data = load_iris()
    input_data = data['data']
    output_data = data['target']
    outputs_name = data['target_names']
    feature_names = list(data['feature_names'])
    feature_1 = [feat[feature_names.index('sepal length (cm)')] for feat in input_data]
    feature_2 = [feat[feature_names.index('sepal width (cm)')] for feat in input_data]
    feature_3 = [feat[feature_names.index('petal length (cm)')] for feat in input_data]
    feature_4 = [feat[feature_names.index('petal width (cm)')] for feat in input_data]
    input_data = [[feat[feature_names.index('sepal length (cm)')],
                   feat[feature_names.index('sepal width (cm)')],
                   feat[feature_names.index('petal length (cm)')],
                   feat[feature_names.index('petal width (cm)')]] for feat in input_data]
    return input_data, output_data, outputs_name, feature_1, feature_2, feature_3, feature_4, feature_names


def load_data_digit():
    data = load_digits()
    input_data = data.images
    output_data = data['target']
    outputs_name = data['target_names']

    #shuffle the original data
    no_data=len(input_data)
    permutation=np.random.permutation(no_data)
    input_data=input_data[permutation]
    output_data=output_data[permutation]
    return input_data, output_data, outputs_name


def plot_histogram_feature(feature, variableName):
    plt.hist(feature, 10)
    plt.title('Histogram of ' + variableName)
    plt.show()


def plot_histogram_data(output_data, outputs_name, title):
    plt.hist(output_data, rwidth=0.8)
    plt.title('Histogram of ' + title)
    plt.xticks(np.arange(len(outputs_name)), outputs_name)
    plt.show()


def train_and_test(input_data, output_data):
    #np.random.seed(5)
    indexes = [i for i in range(len(input_data))]
    train_sample = np.random.choice(indexes, int(0.8 * len(input_data)), replace=False)
    test_sample = [i for i in indexes if i not in train_sample]
    train_inputs = [input_data[i] for i in train_sample]
    train_outputs = [output_data[i] for i in train_sample]
    test_inputs = [input_data[i] for i in test_sample]
    test_outputs = [output_data[i] for i in test_sample]
    return train_inputs, train_outputs, test_inputs, test_outputs


def normalisation(train_data, test_data):
    scaler = StandardScaler()
    if not isinstance(train_data[0], list):
        trainData = [[d] for d in train_data]
        testData = [[d] for d in test_data]
        scaler.fit(trainData)
        normalisedTrainData = scaler.transform(trainData)
        normalisedTestData = scaler.transform(testData)
        normalisedTrainData = [el[0] for el in normalisedTrainData]
        normalisedTestData = [el[0] for el in normalisedTestData]
    else:
        scaler.fit(train_data)
        normalisedTrainData = scaler.transform(train_data)
        normalisedTestData = scaler.transform(test_data)
    return normalisedTrainData, normalisedTestData


def classifier_by_tool(train_inputs, train_outputs, test_inputs, test_outputs):
    classifier = neural_network.MLPClassifier(hidden_layer_sizes=(5,), activation='relu', max_iter=1000, solver='sgd',
                                              verbose=0, random_state=1, learning_rate_init=.1)
    classifier.fit(train_inputs, train_outputs)
    computed_outputs = classifier.predict(test_inputs)
    print('Accuracy by tool:', classifier.score(test_inputs, test_outputs))
    return computed_outputs


def classifier_by_me(train_inputs, train_outputs, test_inputs):
    classifier = ANN.NeuralNetwork(hidden_layer_size=10)
    classifier.fit(np.array(train_inputs), np.array(train_outputs))
    computed_outputs = classifier.predict(test_inputs)
    return computed_outputs


def evaluate(test_outputs, computed_labels, output_names):
    confusion_matrix_calculated = confusion_matrix(test_outputs, computed_labels)
    accuracy = sum([confusion_matrix_calculated[i][i] for i in range(len(output_names))]) / len(test_outputs)
    precision = {}
    recall = {}
    for i in range(len(output_names)):
        precision[output_names[i]] = confusion_matrix_calculated[i][i] / sum([confusion_matrix_calculated[j][i]
                                                                         for j in range(len(output_names))])
        recall[output_names[i]] = confusion_matrix_calculated[i][i] / sum([confusion_matrix_calculated[i][j]
                                                                        for j in range(len(output_names))])
    print('Accuracy by me: ', accuracy)
    print('Precision by me: ', precision)  # TP/TP+FP - cate din cele gasite sunt relevante
    print('Recall by me: ', recall)  # TP/TP+FN - cate  relevante au fost gasite
    return confusion_matrix_calculated


def plotConfusionMatrix(cm, class_names, title):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix ' + title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    text_format = 'd'
    thresh = cm.max() / 2.
    for row, column in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(column, row, format(cm[row, column], text_format),
                 horizontalalignment='center',
                 color='white' if cm[row, column] > thresh else 'black')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def flatten_data(train_data, test_data):
    #the image is represented as a two-dimensional array of pixel values
    #by flattening the image, we can transform it into a one-dimensional array of pixel values, making it easier to feed into a neural network
    def flatten(mat):
        x = []
        for line in mat:
            for el in line:
                x.append(el)
        return x
    train_data = [flatten(data) for data in train_data]
    test_data = [flatten(data) for data in test_data]
    return train_data, test_data


if __name__ == '__main__':
    print("IRIS")
    inputs, outputs, output_names, feature1, feature2, feature3, feature4, featureNames = load_data_flowers()
    # plot_histogram_feature(feature1, featureNames[0])
    # plot_histogram_feature(feature2, featureNames[1])
    # plot_histogram_feature(feature3, featureNames[2])
    # plot_histogram_feature(feature4, featureNames[3])
    plot_histogram_data(outputs, output_names, 'Iris')
    train_inputs, train_outputs, test_inputs, test_outputs = train_and_test(inputs, outputs)
    train_inputs, test_inputs = normalisation(train_inputs, test_inputs)
    computed_outputs = classifier_by_tool(train_inputs, train_outputs, test_inputs, test_outputs)
    print('Computed by tool: ', list(computed_outputs))
    print('Real:             ', test_outputs)
    print()
    computed_outputs_by_me = classifier_by_me(train_inputs, train_outputs, test_inputs)
    print('Computed by me:', computed_outputs_by_me)
    print('Real:          ', test_outputs)
    confusion_matrix_by_me = evaluate(np.array(test_outputs), np.array(computed_outputs_by_me), output_names)
    plotConfusionMatrix(confusion_matrix_by_me, output_names, "Iris classification by me")

    print()
    print('DIGITS')
    inputs, outputs, output_names = load_data_digit()
    plot_histogram_data(outputs, output_names, 'Digits')
    train_inputs, train_outputs, test_inputs, test_outputs = train_and_test(inputs, outputs)
    train_inputs, test_inputs = flatten_data(train_inputs, test_inputs)
    train_inputs, test_inputs = normalisation(train_inputs, test_inputs)
    computed_outputs = classifier_by_tool(train_inputs, train_outputs, test_inputs, test_outputs)
    print('Computed by tool: ', list(computed_outputs))
    print('Real:             ', test_outputs)
    print()
    computed_outputs_by_me = classifier_by_me(train_inputs, train_outputs, test_inputs)
    print('Computed by me: ', computed_outputs_by_me)
    print('Real:           ', test_outputs)
    confusion_matrix_by_me = evaluate(np.array(test_outputs), np.array(computed_outputs_by_me), output_names)
    plotConfusionMatrix(confusion_matrix_by_me, output_names, "Digits classification by me")