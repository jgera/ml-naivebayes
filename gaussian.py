import math


def mean(fv):
    '''Calculate the mean of a feauture vector.'''
    return float(sum(fv)) / len(fv)


def mvue(fv, mu=None):
    '''Minimum variance unbiased estimation of a feature vector.'''
    mu = mean(fv) if mu is None else float(mu)
    return sum([math.pow(v - mu, 2) for v in fv]) / (len(fv) - 1)


def density(mu, sd, stddev=True):
    '''Return a Gaussian PDF with the given mean and variance or std deviation.

    If stddev is true, treat sd as the standard deviation, otherwise treat it
    as the variance.

    '''
    mu = float(mu)
    var = math.pow(sd, 2) if stddev else sd
    if var == 0.0:
        raise ZeroVarianceError
    return lambda x: math.pow(math.e, -1 * math.pow(x - mu, 2) / 2 * var) / math.sqrt(2 * math.pi * var)


def model(fv, mu=None, sd=None):
    '''Produce a Gaussian PDF for the feature vector.

    mu is the estimated mean.
    If the mu is None, use the mean of the feature values.

    sd is the standard deviation.
    If the sd is None, use MVUE to estimate variance.

    '''
    mu = mean(fv) if mu is None else mu
    sd = mvue(fv, mu) if sd is None else sd
    return density(mu, sd, not (sd is None))

class ZeroVarianceError(ValueError):
    pass
