import numpy as np


class NeuralNetwork:

    def __init__(self, hidden_layer_size=10, max_iter=10000, learning_rate=0.0001):
        self.__weights = []
        self.__hidden_layer_size = hidden_layer_size
        self.__max_iter = max_iter
        self.__learning_rate = learning_rate

    def __softmax(self, x):
        # activation function
        exp_vector = np.exp(x)
        return exp_vector / exp_vector.sum(axis=1, keepdims=True)

    def __sigmoid(self, x):
        # activation function (one vs all)
        return 1 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, x):
        # a;ways non-negative
        return self.__sigmoid(x) * (1 - self.__sigmoid(x))

    def fit(self, x, y):
        # determine number of features and outputs
        no_features = len(x[0])
        no_outputs = len(set(y))

        # one-hot encode target variable beacuse we are working with labels
        new_y = np.zeros((len(y), no_outputs))  # matrix
        for i in range(len(y)):
            new_y[i, y[i]] = 1
        y = new_y  # matrix

        # initialize weights and biases
        weight_input_hidden = np.random.rand(no_features, self.__hidden_layer_size)
        coefficient_input_hidden = np.random.randn(self.__hidden_layer_size)
        weight_hidden_output = np.random.rand(self.__hidden_layer_size, no_outputs) # matrix [number_of_hidden, number_of_outputs]
        coefficient_hidden_output = np.random.randn(no_outputs)

        # train the model using backpropagation
        for epoch in range(self.__max_iter):
            # feed forward
            y_input_hidden = np.dot(x,weight_input_hidden) + coefficient_input_hidden  # matrix [number_of_samples, hidden_layer_size]
            y_input_hidden_sigmoid = self.__sigmoid(y_input_hidden)
            y_output = np.dot(y_input_hidden_sigmoid, weight_hidden_output) + coefficient_hidden_output
            y_output_softmax = self.__softmax(y_output)  # abordare softmax

            # compute error
            error = y_output_softmax - y # matrix [number_of_samples, number_of_outputs]

            # backpropagation
            error_weight_hidden_output = np.dot(y_input_hidden_sigmoid.T, error)
            error_coefficient_hidden_output = error
            error_derivatice_activation_hidden = np.dot(error, weight_hidden_output.T)
            #for backpropagation, we need to compute the rate of change of the output of the sigmoid function, which is given by its derivative
            derivative_sigmoid_y_input_hidden = self.__sigmoid_derivative(
                y_input_hidden)  # always the same sign as the error
            output_delta = x
            error_weight_input_hidden = np.dot(output_delta.T,
                                               derivative_sigmoid_y_input_hidden * error_derivatice_activation_hidden)
            error_coefficient_input_hidden = error_derivatice_activation_hidden * derivative_sigmoid_y_input_hidden

            # update weights and coefficients delta rule (using gradient descent)
            weight_input_hidden -= self.__learning_rate * error_weight_input_hidden
            coefficient_input_hidden -= self.__learning_rate * error_coefficient_input_hidden.sum(axis=0)
            weight_hidden_output -= self.__learning_rate * error_weight_hidden_output
            coefficient_hidden_output -= self.__learning_rate * error_coefficient_hidden_output.sum(axis=0)

        # store learned weights and biases
        self.__weights = [weight_input_hidden, coefficient_input_hidden, weight_hidden_output,
                          coefficient_hidden_output]

    def predict(self, x):
        weight_input_hidden, coefficient_input_hidden, weight_hidden_output, coefficient_hidden_output = self.__weights
        y_input_hidden = np.dot(x, weight_input_hidden) + coefficient_input_hidden
        y_input_hidden_sigmoid = self.__sigmoid(y_input_hidden)
        y_output = np.dot(y_input_hidden_sigmoid, weight_hidden_output) + coefficient_hidden_output
        y_output_softmax = self.__softmax(y_output)
        #computed_output = [list(output).index(max(output)) for output in y_output_softmax]
        computed_output = []
        for output in y_output_softmax:
            row = list(output)
            max_index = row.index(max(row))
            computed_output.append(max_index)
        return computed_output
