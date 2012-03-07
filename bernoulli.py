import math


def mean(fv):
    '''Calculate the mean of a feauture vector.'''
    return float(sum(fv)) / len(fv)


def randomvariable(th):
    '''Return a Bernoulli random variable for the threshold.'''
    return lambda x: th < x


def distribution(fv, rv):
    '''Return a Bernoulli PMF for the feature vector and random variable.'''
    successCt = sum([rv(f) for f in fv])
    prSuccess = float(          successCt + 1) / (len(fv) + 2) # Laplace
    prFailure = float(len(fv) - successCt + 1) / (len(fv) + 2) # Smoothing
    return lambda x: prSuccess if rv(x) else prFailure


def model(fv, th=None):
    '''Produce a Bernoulli PMF for the feature vector.

    If the threshold is not specified, use the mean of the feature values.

    '''
    rv = randomvariable(mean(fv) if th is None else th)
    return distribution(fv, rv)
