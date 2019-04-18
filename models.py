import sklearn.ensemble.AdaBoostClassifier as AdaBoostClassifier
import sklearn.svm.SVR as SVR

def adaboost_svm():
    svm = SVR(
        kernel='rbf',
        degree=3,
        gamma='auto_depricated',
        coef0=0.0,
        tol=0.001,
        C=1.0,
        epsilon=0.1,
        shrinking=True,
        cache_size=200,
        verbose=False,
        max_iter=-1
    )

    adaboost = AdaBoostClassifier(
        base_estimator=svm
    )

    return adaboost
