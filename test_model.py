from evaluate import print_results
from evaluate import roc_curve
from models import adaboost_svm
from sklearn.model_selection import train_test_split
import numpy as np


# TODO
def n_split(x, y, n):
    """Yield successive n-sized chunks from x."""
    for i in range(0, len(l), n):
        yield l[i:i + n]



if __name__ == '__main__':

    X = np.loadtxt(open('data/training_data.csv'), delimiter=',')
    y = np.loadtxt(open('data/training_labels.csv'), delimiter=',')




    model = adaboost_svm()



    model.fit(X, y)



