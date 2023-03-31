import numpy as np
import matplotlib.pyplot as plt


def mse(imageA, imageB):
    """Calculate the 'Mean Squared Error' between the two images."""
    return np.mean((imageA - imageB) ** 2)


def normalize(trainx):
    """Normalize and returns the calculated means and stds for each feature"""
    trainxn = trainx.copy()
    if trainx.ndim > 1:
        means = np.zeros((trainx.shape[1], 1))
        stds = np.zeros((trainx.shape[1], 1))
        for n in range(trainx.shape[1]):
            means[n, ] = np.mean(trainxn[:, n])
            stds[n, ] = np.std(trainxn[:, n])
            trainxn[:, n] = (trainxn[:, n] - means[n, ]) / (stds[n, ])
    else:
        means = np.mean(trainxn)
        stds = np.std(trainxn)
        trainxn = (trainxn - means) / stds
    return trainxn, means, stds


def applynormalize(testx, means, stds):
    """Apply normalization based on previous calculated means and stds"""
    testxn = testx.copy()
    if testxn.ndim > 1:
        for n in range(testx.shape[1]):
            testxn[:, n] = (testxn[:, n] - means[n, ]) / (stds[n, ])
    else:
        testxn = (testxn - means) / stds
    return testxn


def reversenormalize(testx, means, stds):
    """Reverse normalization based on previous calculated means and stds"""
    testxn = testx.copy()
    if testxn.ndim > 1:
        for n in range(testx.shape[1]):
            testxn[:, n] = (testxn[:, n] * stds) + means
    else:
        testxn = (testxn * stds) + means

    return testxn


def minMaxScale(trainx):
    """Normalize and returns the calculated max and min of the output"""
    trainxn = trainx.copy()
    maxs = np.max(trainxn)
    mins = np.min(trainxn)
    trainxn = (trainxn - mins) / (maxs - mins) * 10
    return trainxn, maxs, mins


def applyMinMaxScale(testx, maxs, mins):
    """Apply normalization based on previous calculated means and stds"""
    testxn = testx.copy()
    testxn = (testxn - mins) / (maxs - mins) * 10
    return testxn


def reverseMinMaxScale(testx, maxs, mins):
    """Apply normalization based on previous calculated means and stds"""
    testxn = testx.copy()
    if testxn.ndim > 1:
        for n in range(testx.shape[1]):
            testxn[:, n] = (testxn[:, n] * (maxs - mins) / 10) + mins
    else:
        testxn = (testxn * (maxs - mins) / 10) + mins

    return testxn
