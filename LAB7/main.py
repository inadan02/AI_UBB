import csv
import matplotlib.pyplot as plt
import numpy as np

from BGDRegression import tool_univariat, MySGDRegression


def load_all_data(file):
    data = []
    data_names = []
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count != 0:
                data.append(row)
            else:
                data_names = row  # if it's on the first row it's a table header, si it is a label name
            line_count += 1
    return data_names, data


def extract_feature(all_data, data_names, feature_name):
    pos = data_names.index(feature_name)  # wanted feature index
    return [float(data[pos]) for data in all_data]


def scale01(feature):
    # min-max scaling x_new= (x - min(x)) / (max(x) - min(x))
    # range 0-1
    #we know the approximate upper and lower bounds
    #data is approximately uniformly distributed across the range
    minim = min(feature)
    maxim = max(feature)
    feature_scaled_0_1 = [(f - minim) / (maxim - minim) for f in feature]
    return feature_scaled_0_1


def data_normalisation(feature1, feature2):
    # map the data in the same interval: 0-1
    feature1_scaled = scale01(feature1)
    feature2_scaled = scale01(feature2)
    return feature1_scaled, feature2_scaled


def predictionErr(realOutputs, computedOutputs):
    # compute the prediction error
    # mean absolute error
    error = 0
    for t1, t2 in zip(computedOutputs, realOutputs):
        error += abs(t2 - t1)
    error /= len(realOutputs)
    print('MAE error: ', error)


def calculate_y(coefficients, features):
    # modelul liniar de predictie (regresorul): y=w0+w1*x1+w2*x2    (for 2 features)
    y = []
    for elem in features:
        feature1 = elem[0]
        feature2 = elem[1]
        y.append(coefficients[0] + feature1 * coefficients[1] + feature2 * coefficients[2])
    return y


def plot_2d(train_feature1, w0, w1, train_result, title=None):
    noOfPoints = 1000
    xref = []
    val = min(train_feature1)
    step = (max(train_feature1) - min(train_feature1)) / noOfPoints
    for i in range(1, noOfPoints):
        xref.append(val)
        val += step
    yref = [w0 + w1 * el for el in xref]

    plt.plot(train_feature1, train_result, 'ro', label='training data')  # train data are plotted by red and circle sign
    plt.plot(xref, yref, 'b-', label='learnt model')  # model is plotted by a blue line
    # plt.title('train data and the learnt model')
    plt.title(title)
    plt.xlabel('GDP capita')
    plt.ylabel('happiness')
    plt.legend()
    plt.show()


def plot_3d(x1_train, x2_train, y_train, x1_model=None, x2_model=None, y_model=None, x1_test=None, x2_test=None,
            y_test=None, title=None):
    ax = plt.axes(projection='3d')
    if x1_train:
        plt.scatter(x1_train, x2_train, y_train, c='r', marker='o', label='train data')
    if x1_model:
        plt.scatter(x1_model, x2_model, y_model, c='b', marker='_', label='learnt model')
    if x1_test:
        plt.scatter(x1_test, x2_test, y_test, c='g', marker='^', label='test data')
    plt.title(title)
    ax.set_xlabel("gdp capita")
    ax.set_ylabel("freedom")
    ax.set_zlabel("happiness")
    plt.legend()
    plt.show()

def plotDataHistogram(x, variableName):
    n, bins, patches = plt.hist(x, 10)
    plt.title('Histogram of ' + variableName)
    plt.show()

def main():
    file = 'data/v1_world-happiness-report-2017.csv'
    # read the data
    names, data = load_all_data(file)
    input1 = extract_feature(data, names, 'Economy..GDP.per.Capita.')
    input2 = extract_feature(data, names, 'Freedom')
    output = extract_feature(data, names, 'Happiness.Score')

    plotDataHistogram(input1, "capita GDP")
    plotDataHistogram(input2, "freedom")
    plotDataHistogram(output, "Happiness score")

    # split data into training and testing
    indexes = []
    for index in range(len(input1)):
        indexes.append(index)
    # generam indecsii pentru training data
    dimensiune_train_sample = int(0.8 * len(input1))  # 80% din date pentru training
    train_sample_indexes = np.random.choice(indexes, dimensiune_train_sample,
                                            replace=False)  # replace=false=>doar o data poate fi ales
    validation_sample_indexes = []
    for i in indexes:
        if i not in train_sample_indexes:
            validation_sample_indexes.append(i)  # restul 20% ce nu a fost ales e pentru testing
    train_feature1 = [input1[i] for i in train_sample_indexes]
    train_feature2 = [input2[i] for i in train_sample_indexes]
    train_result = [output[i] for i in train_sample_indexes]

    validation_feature1 = [input1[i] for i in validation_sample_indexes]
    validation_feature2 = [input2[i] for i in validation_sample_indexes]
    validation_result = [output[i] for i in validation_sample_indexes]

    # normalizam datele pentru regresia bivariata, pt ca GDP si Freedom pot avea unitati de masura dif
    train_feature1_scaled, train_feature2_scaled = data_normalisation(train_feature1, train_feature2)

    print('univariat cu tool:')
    tool_univariat(train_feature1, train_result)
    print('univariat fara tool:')
    regressor = MySGDRegression()
    xx = [[el] for el in train_feature1]
    regressor.fit(xx, train_result)
    w0, w1 = regressor.intercept_, regressor.coef_[0]
    print('Regressor calculated by me: f(x) = ', w0, ' + ', w1, ' * x')
    # plot results
    plot_2d(train_feature1, w0, w1, train_result, 'capita vs happiness')

    print('multivariat:')
    regressor2 = MySGDRegression()
    regressor2.fit(list(zip(train_feature1_scaled, train_feature2_scaled)), train_result)
    w0_multivariat, w1_multivariat, w2_multivariat = regressor2.intercept_, regressor2.coef_[0], regressor2.coef_[1]
    print('Regressor calculated by me: f(x) = ', w0_multivariat, ' + ', w1_multivariat, ' * x', ' + ', w2_multivariat,
          ' * x2')

    noOfPoints = 10
    xref1 = []
    val = min(train_feature1_scaled)
    step1 = (max(train_feature1_scaled) - min(train_feature1_scaled)) / noOfPoints
    for _ in range(1, noOfPoints):
        for _ in range(1, noOfPoints):
            xref1.append(val)
        val += step1

    xref2 = []
    val = min(train_feature2_scaled)
    step2 = (max(train_feature2_scaled) - min(train_feature2_scaled)) / noOfPoints
    for _ in range(1, noOfPoints):
        aux = val
        for _ in range(1, noOfPoints):
            xref2.append(aux)
            aux += step2
    yref = [w0_multivariat + w1_multivariat * el1 + w2_multivariat * el2 for el1, el2 in zip(xref1, xref2)]
    plot_3d(train_feature1_scaled, train_feature2_scaled, train_result, xref1, xref2, yref, [], [], [],
            'capita vs freedom vs happiness')


main()
