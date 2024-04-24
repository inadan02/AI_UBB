import math
from math import sqrt
import pandas as pd
import os


crtDir = os.getcwd()
file_sports = os.path.join(crtDir, 'data', 'sports.csv')
file_flowers = os.path.join(crtDir, 'data', 'flowers.csv')
sports = pd.read_csv(file_sports)
flowers = pd.read_csv(file_flowers)


def MeanAbsoluteError(real, computed):
    # regression-> sum of absolute difference
    error = 0
    for r, c in zip(real, computed):
        for i in range(len(r)):
            error += abs(r[i] - c[i])
    return error / len(real[0])


def RootMeanSquareError(real, computed):
    # regression-> sum of square difference
    error = 0
    for r, c in zip(real, computed):
        for i in range(len(r)):
            error += (r[i] - c[i]) ** 2
    return sqrt(error / len(real[0]))


print("Pb 1")
print("MAE:", MeanAbsoluteError([sports['Weight'], sports['Waist'], sports['Pulse']],
                                [sports['PredictedWeight'], sports['PredictedWaist'],
                                 sports['PredictedPulse']]))

print("RMSE:", RootMeanSquareError([sports['Weight'], sports['Waist'], sports['Pulse']],
                                   [sports['PredictedWeight'], sports['PredictedWaist'],
                                    sports['PredictedPulse']]))


def classification(real, computed, labels):
    #error of prediction
    accuracy = sum([1 if real[i] == computed[i] else 0 for i in range(0, len(real))]) / len(real)
    precision = {}
    recall = {}
    TP = {}
    FP = {}
    TN = {}
    FN = {}
    for label in labels:
        TP[label] = sum([1 if (real[i] == label and computed[i] == label) else 0 for i in range(len(real))])
        FP[label] = sum([1 if (real[i] != label and computed[i] == label) else 0 for i in range(len(real))])
        TN[label] = sum([1 if (real[i] != label and computed[i] != label) else 0 for i in range(len(real))])
        FN[label] = sum([1 if (real[i] == label and computed[i] != label) else 0 for i in range(len(real))])
        print(label, TP[label], FP[label], TN[label], FN[label])
    for label in labels:
        precision[label] = TP[label] / (TP[label] + FP[label])#cati sunt cu adevarat
        recall[label] = TP[label] / (TP[label] + FN[label])#cati a nimerit algoritmul
    return accuracy, precision, recall


print("\nPb 2")
flowers_types = list(set(flowers['Type']))#label-rile distincte
accuracy, precisions, recalls = classification(flowers['Type'], flowers['PredictedType'],
                                               flowers_types)
print("Accuracy: ", accuracy)
print("Precision for", flowers_types[0], "is: ", precisions[flowers_types[0]])
print("Precision for", flowers_types[1], "is: ", precisions[flowers_types[1]])
print("Precision for", flowers_types[2], "is: ", precisions[flowers_types[2]])
print("Recall for", flowers_types[0], "is: ", recalls[flowers_types[0]])
print("Recall for", flowers_types[1], "is: ", recalls[flowers_types[1]])
print("Recall for", flowers_types[2], "is: ", recalls[flowers_types[2]])


def loss_regression(real, computed):
    #eroare efectiva, costul=>nu mai impart la nr total
    loss = 0.0
    for real_value, computed_value in zip(real, computed):
        for index in range(len(real_value)):
            loss += abs(real_value[index] - computed_value[index])
    return loss


print("\nPb1 extra")
print("Loss regression:", loss_regression([sports['Weight'], sports['Waist'], sports['Pulse']],
                                        [sports['PredictedWeight'], sports['PredictedWaist'],sports['PredictedPulse']]))


def loss_binary_classification(real, computed, positive_class):
    real_outputs=[]
    for label in real:#avem label-uri reale=>codificam
        if label==positive_class:
            real_outputs.append([1,0])
        else:
            real_outputs.append([0, 1])
    no_of_classes = len(set(real))
    sum_cross_entropy = 0.0 #cross entropy
    for i in range(len(real)):
        #local_cross_entropy = - sum([real_outputs[i][j] * math.log(computed[i][j]) for j in range(no_of_classes)]) #log(x)<0, x apartine(0,1) general cross entropy
        local_cross_entropy = sum([-(real_outputs[i][j] * math.log(computed[i][j], 2) + (1 - real_outputs[i][j]) * math.log(1 - computed[i][j], 2))for j in range(no_of_classes)])#binary cross entropy
        sum_cross_entropy += local_cross_entropy
    cross_entropy = sum_cross_entropy / len(real)
    return cross_entropy
    #return real_outputs


print("\nPb2 extra")
real = ['spam', 'spam', 'ham', 'ham', 'spam', 'ham']
computed = [[0.7, 0.3], [0.2, 0.8], [0.4, 0.6], [0.9, 0.1], [0.7, 0.3], [0.4, 0.6]]
print("Loss binary ham:", loss_binary_classification(real, computed, 'ham')) #[ham, spam]
print("Loss binary spam:", loss_binary_classification(real, computed, 'spam')) #[spam. ham]


def loss_multi_class(target_values, raw):
    #softmax function
    expected_values=[]
    for value in raw:
        expected_values.append(math.exp(value))
    sum_expected_values = sum(expected_values)
    outputs=[]
    for value in expected_values:
        outputs.append(value/sum_expected_values)
    cross_entropy = - sum([target_values[j] * math.log(outputs[j]) for j in range(len(target_values))])
    return cross_entropy


print("\nPb3 extra")
print("Loss multi-class", loss_multi_class([0, 0, 1, 0, 0], [0.4, -0.6, 0.1, 1.3, 2.3]))


def loss_multi_label(target_values, raw):
    #sigmoid function
    outputs=[]
    for value in raw:
        sigmoid=1/(1+math.exp(-value))
        outputs.append(sigmoid)
    cross_entropy = - sum([target_values[i] * math.log(outputs[i]) for i in range(len(target_values))])
    return cross_entropy


print("\nPb 4 extra")
print("Loss multi-label:", loss_multi_label([0, 1, 1, 0, 1], [0.4, -0.6, 0.1, 1.3, 2.3]))

