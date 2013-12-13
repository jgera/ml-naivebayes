import math


def mean(fv):
    '''Calculate the mean of a feauture vector.'''
    return float(sum(fv)) / len(fv)


def mvue(fv, mu=None):
    '''Minimum variance unbiased estimation of a feature vector.'''
    mu = mean(fv) if mu is None else float(mu)
    return sum([math.pow(v - mu, 2) for v in fv]) / (len(fv) - 1)


def density(mu, var):
    '''float float -> (num -> float)
    
    Return a Gaussian PDF with the given mean and variance
    
    '''
    if var == 0.0:
        raise ZeroVarianceError
    # constants in the pdf
    # mu = ..
    # var = ..
    sd = math.sqrt(var)
    a = 1 / (sd * math.sqrt(2 * math.pi))
    # ret
    def pdf(x):
        b = -1 * math.pow(x - mu, 2) / (2 * var)
        return a * math.exp(b)
    return pdf


def model(fv):
    '''[num] -> (num -> float)
    
    Produce a Gaussian PDF for the feature vector.

    '''
    mu = mean(fv)
    var = mvue(fv)
    return density(mu, var)

class ZeroVarianceError(ValueError):
    pass
