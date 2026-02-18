import numpy as np
from sklearn import linear_model
from sklearn.cross_decomposition import PLSRegression

def regress(
    X_train,
    Y_train,
    X_test,
    model_type="ridge",
    alpha=0.1,
    n_components=20,
):
    """
    Fit a linear model on (X_train, Y_train) and predict Y on X_test.

    By default, uses Ridge regression (alpha=0.1), but other options
    like PLS are supported via `model_type`.

    Parameters
    ----------
    X_train : array-like, shape (n_samples_train, n_features)
        Training features.
    Y_train : array-like, shape (n_samples_train, n_targets)
        Training targets.
    X_test : array-like, shape (n_samples_test, n_features)
        Test features to predict on.
    model_type : {"ridge", "linear", "lasso", "elasticnet", "pls"}, optional
        Which linear model to use. Default is "ridge".
    alpha : float, optional
        Regularization strength for Ridge/Lasso/ElasticNet. Default is 0.1.
    n_components : int, optional
        Number of components for PLSRegression. Default is 20.

    Returns
    -------
    Y_test_pred : array-like, shape (n_samples_test, n_targets)
        Predicted targets for X_test.
    """

    model_type = model_type.lower()

    if model_type == "ridge":
        clf = linear_model.Ridge(alpha=alpha)
    elif model_type == "linear":
        clf = linear_model.LinearRegression()
    elif model_type == "lasso":
        clf = linear_model.Lasso(alpha=alpha)
    elif model_type == "elasticnet":
        clf = linear_model.ElasticNet(alpha=alpha)
    elif model_type == "pls":
        clf = PLSRegression(n_components=n_components)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")

    clf.fit(X_train, Y_train)
    Y_test_pred = clf.predict(X_test)

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












