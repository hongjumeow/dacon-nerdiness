import numpy as np
from sklearn.metrics import accuracy_score, precision_score

def print_accuracy(pred, gt):
    errors = abs(pred - gt)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

    mape = 100 * errors
    acc = 100 - np.mean(mape)
    print('Accuracy:', round(acc, 2), '%.')

def print_sklearn_scores(pred, gt):
    acc_score = accuracy_score(pred, gt)
    print(acc_score)
    prec_score = precision_score(pred, gt)
    print(prec_score)