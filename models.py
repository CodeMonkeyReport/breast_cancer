from sklearn.ensemble import AdaBoostClassifier
# import sklearn.svm.SVR as SVR
import keras
from keras import layers
from keras import backend

def cnn():
    optimizer = keras.optimizers.RMSprop( lr=0.001, clipvalue=1.0)

    input_layer_one = keras.Input(shape=(None, 12750))
    m = layers.Conv1D( 32, (5,), activation=backend.sigmoid )(input_layer_one)
    # 124, 5
    m = layers.Conv1D( 32, (3,), activation=backend.sigmoid )(m)
    m = layers.AveragePooling1D()(m)

    m = layers.Dense(52, activation=backend.sigmoid)(m)
    m = layers.Dropout(0.5)(m)
    m = layers.Dense(26, activation=backend.sigmoid)(m)

    m = layers.Dense(1 )(m)

    model = keras.models.Model( inputs=input_layer_one, outputs=m )
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')

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
