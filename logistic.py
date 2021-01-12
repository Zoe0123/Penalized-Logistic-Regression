from utils import sigmoid

import numpy as np


def logistic_predict(weights, data):
    """ Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :return: A vector of probabilities with dimension N x 1, which is the output
    to the classifier.
    """
    # add dummy '1's to data input
    dummy = np.ones((data.shape[0],1), dtype=int)
    data_dummy = np.hstack((data, dummy))

    z = data_dummy @ weights
    y = sigmoid(z)
    return y


def evaluate(targets, y):
    """ Compute evaluation metrics.

    Note: N is the number of examples
          M is the number of features per example

    :param targets: A vector of targets with dimension N x 1.
    :param y: A vector of probabilities with dimension N x 1.
    :return: A tuple (ce, frac_correct)
        WHERE
        ce: (float) Averaged cross entropy
        frac_correct: (float) Fraction of inputs classified correctly
    """

    N = targets.shape[0]

    a = - (targets.T @ np.log(y))

    ones = np.ones((N, 1), dtype=int)
    b = (ones-targets).T @ np.log(ones-y)
    # avg ce = 1/N * sum{-(t * log(y)) - [(1-t) * log(1-y)]}
    ce = ((a - b) / N).item()   

    # count number of correct classification in y (y >= 0.5 is positive classification)
    correct = np.count_nonzero(targets == (y//0.5))
    frac_correct = correct / N

    return ce, frac_correct


def logistic(weights, data, targets, hyperparameters):
    """ Calculate the cost and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :param targets: A vector of targets with dimension N x 1.
    :param hyperparameters: The hyperparameter dictionary.
    :returns: A tuple (f, df, y)
        WHERE
        f: The average of the loss over all data points.
           This is the same as averaged cross entropy.
           This is the objective that we want to minimize.
        df: (M + 1) x 1 vector of derivative of f w.r.t. weights.
        y: N x 1 vector of probabilities.
    """
    y = logistic_predict(weights, data)

    f = evaluate(targets, y)[0]

    N = targets.shape[0]
    dummy = np.ones((N,1), dtype=int)
    data_dummy = np.hstack((data, dummy))

    df = (data_dummy.T @ (y-targets))/N
    return f, df, y


def logistic_pen(weights, data, targets, hyperparameters):
    """ Calculate the cost of penalized logistic regression and its derivatives
    with respect to weights. Also return the predictions.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :param targets: A vector of targets with dimension N x 1.
    :param hyperparameters: The hyperparameter dictionary.
    :returns: A tuple (f, df, y)
        WHERE
        f: The average of the loss over all data points, plus a penalty term.
           This is the objective that we want to minimize.
        df: (M+1) x 1 vector of derivative of f w.r.t. weights.
        y: N x 1 vector of probabilities.
    """
    y = logistic_predict(weights, data)
    
    lambd = hyperparameters["weight_regularization"]
    # add penalty term to ce
    f = evaluate(targets, y)[0] + lambd/2 * sum(np.square(weights[:-1]))   
 
    N = targets.shape[0]
    dummy = np.ones((N,1), dtype=int)
    data_dummy = np.hstack((data, dummy))
    df = (data_dummy.T @ (y-targets))/N + lambd * np.append(weights[:-1], [[0]], axis=0)
    return f, df, y
