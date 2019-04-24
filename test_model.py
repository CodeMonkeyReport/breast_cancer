from evaluate import print_results
from evaluate import roc_curve
from models import adaboost_svm
from models import cnn
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.ensemble import AdaBoostClassifier
# import sklearn.svm.SVR as SVR
import keras
from keras import layers
from keras import backend

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling1D
from keras.layers import Conv1D
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.layers import Reshape
from keras.layers import Activation
from sklearn.preprocessing import normalize


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
    fpr,tpr,thresholds = metrics.roc_curve(real,pred)
    plt.plot(fpr,tpr)
    plt.plot([0,1],[0,1], color='k')
    plt.show()


def cnn():
    optimizer = keras.optimizers.RMSprop( lr=0.0001, clipvalue=1.0)

    input_layer_one = keras.Input(shape=(1250,))
    #m = layers.Reshape((12750, 1))(input_layer_one)
    
    #m = layers.Conv1D( 16, (50,), strides=10, activation=backend.relu )(m)
    # 124, 5
    #m = layers.Conv1D( 16, (20,), strides=10, activation=backend.relu )(m)
    #m = layers.MaxPooling1D(pool_size=50, strides=5)(m)

    #m = layers.Flatten()(input_layer_one)

    m = layers.Dense(64, activation=backend.relu)(input_layer_one)
    m = layers.Dropout(0.4)(m)
    m = layers.Dense(64, activation=backend.relu)(m)
    m = layers.Dropout(0.4)(m)
    m = layers.Dense(1, activation=backend.sigmoid)(m)

    model = keras.models.Model( inputs=input_layer_one, outputs=m )
    model.compile(optimizer=optimizer, loss='binary_crossentropy')

    return model

def adaboost_svm():
    # svm = SVR(
    #     kernel='rbf',
    #     degree=3,
    #     gamma='auto_depricated',
    #     coef0=0.0,
    #     tol=0.001,
    #     C=1.0,
    #     epsilon=0.1,
    #     shrinking=True,
    #     cache_size=200,
    #     verbose=False,
    #     max_iter=-1
    # )

    adaboost = AdaBoostClassifier(
        # base_estimator=svm
    )

    return adaboost

def run_nn(x,y):
  # Set up the neural network
  epochs = 35
  learning_rate = 0.001
  verbose = 1

  # Set up a kfold object
  seed = 10
  np.random.seed(seed)
  kfold = KFold(n_splits=10, shuffle=True,random_state=seed)
  
  
  real = []
  prediction = []
  # Loop through the folded tests
  for train, test in kfold.split(x):
    # Make a new model
    model = richard_cnn2()
    
    positive_count = np.sum(y[train])
    negative_count = len(y[train]) - np.sum(y[train])
    
    class_weight = {0: 1,
                    1: negative_count/positive_count }
    # Fit the model
    model.fit(x[train], y[train], epochs=epochs, verbose=verbose, class_weight=class_weight)
    
    pred = model.predict(x[test])

    real.extend(y[test])
    prediction.extend(pred.flatten())


  # Printout average of scores
  print_results(prediction,real)
  
  # Print the ROC Curve
  roc_curve(prediction,real)
  
  # Return the real and predictions
  return [real,prediction]

def richard_cnn2():
    # Convert class vectors to binary class matrices.
    model = Sequential()
    model.add(Reshape((1, 1616, 1), input_shape=(1616,)))
    model.add(Conv2D(128, (1, 125), strides=125, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    return model



if __name__ == '__main__':
    X = np.loadtxt(open('data/training_data.csv'), delimiter=',')
    y = np.loadtxt(open('data/training_labels.csv'), delimiter=',')
    val = np.loadtxt(open('data/validation_data.csv'), delimiter=',')
    val_y = np.loadtxt(open('data/validation_labels.csv'), delimiter=',')

    #x = normalize(x)
    space = PCA()

    X_full = np.concatenate((X, val))
    y_full = np.concatenate((y, val_y))
    
    X_full_normal = normalize(X_full)
    space.fit(X_full_normal)
    X_reduced = space.transform(X_full_normal)
    
    

    (real, pred) = run_nn(X_reduced, y_full)


    print_results(pred, real)

    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.1)
    #X_res = X_train.reshape((X_train.shape[0], 12750, 1))
    val_red = space.transform(val)
    model = cnn()
    scores = cross_val_score(model, X_reduced, y)
    #model.fit(X_train, y_train, epochs=10)
    #pred = model.predict(X_test)
    
    roc_curve(pred, y_test)
    np.savetxt("predictions_cnn.csv", pred)

