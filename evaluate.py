import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

def print_results(pred,real):
    # Show the test data's result
    pred = np.round(pred)
    
    print("Validation set results:")
    print("------------------------------------------------")
    print(metrics.confusion_matrix(real, pred))
    print('average auc  ', metrics.roc_auc_score(real,pred))
    print('kappa:       ', metrics.cohen_kappa_score(real, pred))
    print('accuracy:    ', metrics.accuracy_score(real, pred))
    print('recall:      ', metrics.recall_score(real, pred, average=None))
    print('precision:   ', metrics.precision_score(real, pred, average=None))
    print('f_1:         ', metrics.f1_score(real, pred, average=None))

def roc_curve(pred,real):
    tpr,fpr,thresholds = metrics.roc_curve(real,pred)
    plt.plot(fpr,tpr)
    plt.show()


if __name__ == '__main__':
    predictions = np.random.uniform(low=0.0, high=1.0, size=240) #np.loadtxt(open('prediction.csv','rb'), delimiter=',')
    print(predictions)
    real = np.loadtxt(open('validation_labels.csv','rb'), delimiter=',')
    print(real)

    print_results(predictions,real)
    roc_curve(predictions,real)