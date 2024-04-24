import csv
from math import sqrt

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error


def load_data(filename, input_features, output_feature):
    file = pd.read_csv(filename)
    inputs = []
    for feature in input_features:
        inputs.append([value for value in file[feature]])
    output = [value for value in file[output_feature]]
    # print("input",inputs)
    # print("out", output_feature)
    return inputs, output


def clean_data(input_features, output_feature):
    # removes any rows that contain missing values
    matrix = []
    for index, elems in enumerate(zip(input_features[0], input_features[1])):  # tuplu index, elem
        first, second = elems
        if pd.isna(first) or pd.isna(second):
            input_features[0].pop(index)
            input_features[1].pop(index)
            output_feature.pop(index)
    return input_features, output_feature


def train_and_test(features, result):
    # In this step we will split our dataset into training and testing subsets (in proportion 80/20%)
    np.random.seed(5)
    indexes = [i for i in range(
        len(result))]  # we need to generate the random indexes based on result and use those same indexes to select the corresponding features.
    train_sample_indexes = np.random.choice(indexes, int(0.8 * len(result)), replace=False)
    validation_sample_indexes = [i for i in indexes if i not in train_sample_indexes]
    train_features = []
    validation_features = []
    for feature in features:
        train_features.append([feature[i] for i in train_sample_indexes])
        validation_features.append([feature[i] for i in validation_sample_indexes])
    train_result = [result[i] for i in train_sample_indexes]
    validation_result = [result[i] for i in validation_sample_indexes]
    return train_features, train_result, validation_features, validation_result


def linear_regression_by_tool(train_features, train_result):
    xx = [[x, y] for x, y in zip(train_features[0], train_features[1])]
    regressor = linear_model.LinearRegression()
    regressor.fit(xx, train_result)
    return regressor


def my_linear_regression(features, result):
    # (XT * X)^(-1)*(XT)*Y
    X = [[1] * len(features[0])] + features  # se adauga o linie de 1 precum in functia de regresie din biblioteca (in matricea originala se adauga o col de 1, insa aici avem nevoie de transpusa deci adauga o linie)
    XTX = []
    for row1 in X:
        #iterating through the rows of the transpose of a matrix is equivalent to iterating through the columns of the original matrix
        line = []
        for row2 in X:
            line.append(sum([x * y for x, y in zip(row1, row2)]))
        XTX.append(line)
    XTX_inverse = matrix_inverse(XTX)
    XTX_inverse_XT = []
    for row in XTX_inverse:
        line = []
        for i in matrix_transpose(X):
            line.append(sum([x * y for x, y in zip(row, i)]))
        XTX_inverse_XT.append(line)
    XTX_inverse_XTY = []
    for row in XTX_inverse_XT:
        XTX_inverse_XTY.append(sum([x * y for x, y in zip(row, result)]))
    return XTX_inverse_XTY


def matrix_minor(matrix, i, j):
    return [col[:j] + col[j + 1:] for col in (matrix[:i] + matrix[i + 1:])]


def matrix_determinant(matrix):
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    determinant = 0
    for column in range(len(matrix)):
        determinant += ((-1) ** column) * matrix[0][column] * matrix_determinant(matrix_minor(matrix, 0, column))
    return determinant


def matrix_transpose(matrix):
    #return list(map(list, zip(*matrix)))
    transpose = []
    for i in range(len(matrix[0])):#coloane
        line = []
        for j in range(len(matrix)):#linii
            line.append(matrix[j][i])
        transpose.append(line)
    return transpose


def matrix_inverse(matrix):  # X* / determinant; X* = (-1) ** (r + c) * XT
    determinant = matrix_determinant(matrix)
    if len(matrix) == 2:
        return [[matrix[1][1] / determinant, -1 * matrix[0][1] / determinant],
                [-1 * matrix[1][0] / determinant, matrix[0][0] / determinant]]
    inverse_matrix = []
    for row in range(len(matrix)):
        matrix_row = []
        for column in range(len(matrix)):
            minor = matrix_minor(matrix, row, column)
            matrix_row.append(((-1) ** (row + column)) * matrix_determinant(minor))
        inverse_matrix.append(matrix_row)
    inverse_matrix = matrix_transpose(inverse_matrix)
    for row in range(len(inverse_matrix)):
        for column in range(len(inverse_matrix)):
            inverse_matrix[row][column] = inverse_matrix[row][column] / determinant
    return inverse_matrix


def calculate_y(coefficients, features):
    y = []
    for feature1, feature2 in zip(*features):
        y.append(coefficients[0] + feature1 * coefficients[1] + feature2 * coefficients[2])
    return y


def prediction_error(computed_output, validation_output):
    # mean squared error
    error = 0.0
    for t1, t2 in zip(computed_output, validation_output):
        error += (t1 - t2) ** 2
    error = error / len(validation_output)
    print('Prediction error my MSE: ', error)


def MeanAbsoluteError(computed_output, validation_output):
    # regression-> sum of absolute difference
    error = 0
    for c, v in zip(computed_output, validation_output):
        error += abs(c - v)
    error = error / len(validation_output)
    print('Prediction error my MAE: ', error)


if __name__ == '__main__':
    file_v1 = 'data/v1_world-happiness-report-2017.csv'
    file_v2 = 'data/v2_world-happiness-report-2017.csv'
    file_v3 = 'data/v3_world-happiness-report-2017.csv'
    inputs, output = load_data(file_v3, ['Economy..GDP.per.Capita.', 'Freedom'], 'Happiness.Score')
    inputs, output = clean_data(inputs, output)

    train_inputs, train_outputs, validation_inputs, validation_outputs = train_and_test(inputs, output)
    regressor = linear_regression_by_tool(train_inputs, train_outputs)
    print('Linear regressor calculated by tool:', [] + [regressor.intercept_] + list(regressor.coef_))
    train_regressor = my_linear_regression(train_inputs, train_outputs)
    print('Linear regressor calculated by me:', train_regressor)
    prediction_error(calculate_y(train_regressor, validation_inputs), validation_outputs)
    print("Prediction error tool:",
          mean_squared_error(calculate_y(train_regressor, validation_inputs), validation_outputs))
    print("Prediction error tool mean absolut error:",
          mean_absolute_error(calculate_y(train_regressor, validation_inputs), validation_outputs))
    MeanAbsoluteError(calculate_y(train_regressor, validation_inputs), validation_outputs)
    print("y=", calculate_y(train_regressor, validation_inputs))
