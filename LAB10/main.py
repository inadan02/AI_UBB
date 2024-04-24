import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import neural_network
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import completeness_score
from KNN import MyKNN
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


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


def plot_data_flowers(input_data, output_data, feature_names):
    sns.scatterplot(x=[X[2] for X in input_data],
                    y=[X[3] for X in input_data],
                    hue=output_data,
                    palette="deep",
                    legend=None,
                    s=100)
    plt.xlabel(feature_names[2])
    plt.ylabel(feature_names[3])
    plt.title("All flowers initial data")
    plt.show()


def load_data_spam(filename):
    file = pd.read_csv(filename)
    input_data = [value for value in file["emailText"]]
    output_data = [value for value in file["emailType"]]
    label_names = list(set(output_data))
    return input_data, output_data, label_names


def load_data_emotions(filename):
    file = pd.read_csv(filename)
    input_data = [value for value in file["Text"]]
    output_data = [value for value in file["Sentiment"]]
    label_names = list(set(output_data))
    return input_data, output_data, label_names


def train_and_test(input_data, output_data):
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


def extract_features_bag_of_words(train_inputs, test_inputs, max_features):  # bag of words
    vec = CountVectorizer(max_features=max_features)
    train_features = vec.fit_transform(train_inputs)
    test_features = vec.transform(test_inputs)
    return train_features.toarray(), test_features.toarray()


def extract_features_tf_idf(train_inputs, test_inputs, max_features):  # tf-idf
    vec = TfidfVectorizer(max_features=max_features)
    train_features = vec.fit_transform(train_inputs)
    test_features = vec.fit_transform(test_inputs)
    return train_features.toarray(), test_features.toarray()

def extract_features_hashing(train_inputs, test_inputs, n_features):  # HASHING - uses a hash function to map words directly to a fixed number of indices (hash buckets)
    # instead of maintaining a vocabulary, hashing directly converts words into their hashed representations
    vec = HashingVectorizer(n_features=n_features)
    train_features = vec.fit_transform(train_inputs)
    test_features = vec.fit_transform(test_inputs)
    return train_features.toarray(), test_features.toarray()


def predict_by_tool_unsupervised(train_features, test_features, label_names, classes):
    unsupervisedClassifier = KMeans(n_clusters=classes, random_state=0, n_init="auto")
    unsupervisedClassifier.fit(train_features)
    computed_indexes = unsupervisedClassifier.predict(test_features)
    computed_outputs = [label_names[value] for value in computed_indexes]
    return computed_outputs


def predict_by_me_unsupervised(train_features, test_features, label_names, classes):
    my_unsupervised_classifier = MyKNN(n_clusters=classes)
    my_unsupervised_classifier.fit(train_features)
    my_centroids, computed_indexes = my_unsupervised_classifier.evaluate(test_features)
    computed_outputs = [label_names[value] for value in computed_indexes]
    return computed_outputs, my_centroids, computed_indexes


def predict_supervised(train_inputs, train_outputs, test_inputs):
    classifier = neural_network.MLPClassifier(hidden_layer_sizes=(5,), activation='relu', max_iter=10000,
                                              solver='sgd',
                                              verbose=0, random_state=1, learning_rate_init=.01)
    classifier.fit(train_inputs, train_outputs)#supervised beacause the expected output labels are also given->train_outputs
    computed_outputs = classifier.predict(test_inputs)
    return computed_outputs



def plot_result_flowers(test_inputs, test_outputs, centroids, classification):
    sns.scatterplot(x=[X[2] for X in test_inputs],
                    y=[X[3] for X in test_inputs],
                    hue=test_outputs,
                    style=classification,
                    palette="deep",
                    legend=None,
                    s=100)
    plt.plot([x[2] for x in centroids],
             [y[3] for y in centroids],
             'k+',
             markersize=10)
    plt.title("Test data flowers classification")
    plt.show()


if __name__ == '__main__':

    print("IRIS")
    inputs, outputs, label_names, feature1, feature2, feature3, feature4, featureNames = load_data_flowers()
    plot_data_flowers(inputs, outputs, featureNames)
    train_inputs, train_outputs, test_inputs, test_outputs = train_and_test(inputs, outputs)
    train_inputs, test_inputs = normalisation(train_inputs, test_inputs)
    computed_output = predict_by_tool_unsupervised(train_inputs, test_inputs, label_names, len(set(label_names)))
    print('Completeness score by tool:', completeness_score(test_outputs, computed_output))
    computed_output, centroids, computed_indexes = \
        predict_by_me_unsupervised(train_inputs, test_inputs, label_names, len(set(label_names)))
    print('Completeness score by me:', completeness_score(test_outputs, computed_output))
    plot_result_flowers(test_inputs, test_outputs, centroids, computed_indexes)

    print("\nSPAM")
    inputs, outputs, label_names = load_data_spam('data/spam.csv')
    train_inputs, train_outputs, test_inputs, test_outputs = train_and_test(inputs, outputs)
    train_features, test_features = extract_features_bag_of_words(train_inputs, test_inputs, 1000)
    # train_features, test_features = extract_features_tf_idf(train_inputs, test_inputs, 1000)
    #train_features, test_features = extract_features_hashing(train_inputs, test_inputs, 2**10)
    computed_outputs = predict_by_tool_unsupervised(train_features, test_features, label_names, len(set(label_names)))
    my_computed_outputs, centroids, computed_indexes = \
        predict_by_me_unsupervised(train_features, test_features, label_names, len(set(label_names)))
    inverse_test_outputs = ['spam' if elem == 'ham' else 'ham' for elem in test_outputs]
    accuracy_by_tool = accuracy_score(test_outputs, computed_outputs)
    accuracy_by_tool_inverse = accuracy_score(inverse_test_outputs, computed_outputs)
    print('Accuracy score by tool:', max(accuracy_by_tool, accuracy_by_tool_inverse))
    accuracy_by_me = accuracy_score(test_outputs, my_computed_outputs)
    accuracy_by_me_inverse = accuracy_score(inverse_test_outputs, my_computed_outputs)
    print('Accuracy score by me:', max(accuracy_by_me, accuracy_by_me_inverse))
    print('Output computed:  ', computed_outputs)
    #print('Output computed by me:    ', my_computed_outputs)
    print('Real output:              ', test_outputs)

    print("\nEMOTIONS")
    inputs, outputs, label_names = load_data_emotions('data/reviews_mixed.csv')
    train_inputs, train_outputs, test_inputs, test_outputs = train_and_test(inputs, outputs)
    train_features, test_features = extract_features_bag_of_words(train_inputs, test_inputs, 1000)
    #train_features, test_features = extract_features_tf_idf(train_inputs, test_inputs, 1000)
    # train_features, test_features = extract_features_hashing(train_inputs, test_inputs, 2**10)
    computed_outputs = predict_by_tool_unsupervised(train_features, test_features, label_names, len(set(label_names)))
    my_computed_outputs, centroids, computed_indexes = predict_by_me_unsupervised(train_features, test_features, label_names, len(set(label_names)))
    supervised_output = predict_supervised(train_features, train_outputs, test_features)
    inverse_test_outputs = ['negative' if element == 'positive' else 'positive' for element in test_outputs]
    accuracy_by_tool = accuracy_score(test_outputs, computed_outputs)
    accuracy_by_tool_inverse = accuracy_score(inverse_test_outputs, computed_outputs)
    print('Accuracy score by tool:', max(accuracy_by_tool, accuracy_by_tool_inverse))
    accuracy_by_me = accuracy_score(test_outputs, my_computed_outputs)
    accuracy_by_me_inverse = accuracy_score(inverse_test_outputs, my_computed_outputs)
    print('Accuracy score by me:', max(accuracy_by_me, accuracy_by_me_inverse))
    print('Accuracy score supervised:', accuracy_score(test_outputs, supervised_output))
    print('Output computed:  ', computed_outputs)
    #print('Output computed by me:    ', my_computed_outputs)
    print('Output for supervised:    ', list(supervised_output))
    print('Real output:              ', test_outputs)