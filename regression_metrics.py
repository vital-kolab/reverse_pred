import numpy as np
from sklearn import linear_model

def ridge_regress(X_train, Y_train, X_test, model=None, monkey=None, fold=None):
    """
    Parameters
    ----------
    X_train : TYPE
        DESCRIPTION.
    Y_train : TYPE
        DESCRIPTION.
    X_test : TYPE
        DESCRIPTION.

    Returns
    -------
    Y_test_pred : TYPE
        DESCRIPTION.

    """
    
    clf = linear_model.Ridge(alpha=0.1)
    clf.fit(X_train, Y_train)
    Y_test_pred = clf.predict(X_test)

    if model is not None:
        # Save the weights for later use
        np.save(f'./results_for_figures/model2monkey/{model}_to_{monkey}_ridge_weights_{fold}.npy', clf.coef_)    

    return Y_test_pred

def get_train_test_indices(totalIndices, nrfolds=10, foldnumber=0, seed=1):
    """


    Parameters
    ----------
    totalIndices : TYPE
        DESCRIPTION.
    nrfolds : TYPE, optional
        DESCRIPTION. The default is 10.
    foldnumber : TYPE, optional
        DESCRIPTION. The default is 0.
    seed : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    train_indices : TYPE
        DESCRIPTION.
    test_indices : TYPE
        DESCRIPTION.

    """

    np.random.seed(seed)
    inds = np.arange(totalIndices)
    np.random.shuffle(inds)
    splits = np.array_split(inds, nrfolds)
    test_indices = inds[np.isin(inds, splits[foldnumber])]
    train_indices = inds[np.logical_not(np.isin(inds, test_indices))]
    return train_indices, test_indices


def main():
    if __name__ == "__main__":
        main()












