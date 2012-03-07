import math
import collections


def mean(fv):
    '''Calculate the mean of a feauture vector.'''
    return float(sum(fv)) / len(fv)


def randomvariable(dividers):
    '''Return a histogram random variable which chooses buckets for data.'''
    dividers.sort()
    def fn(x):
        for i, d in enumerate(dividers):
            if x <= d:
                return i
        return len(dividers)
    return fn


def distribution(fv, rv, numclasses):
    '''Return a histogram PMF for the feature vector and random variable.'''
    seen = collections.defaultdict(lambda: 0)
    for f in fv:
        seen[rv(f)] += 1
    return lambda x: (seen[rv(x)] + 1.0) / (len(fv) + numclasses) # Laplace Smoothing


def model(fv, dividers=None):
    '''Produce a histrogram PMF for the feature vector.

    If dividers is None, will make four buckets using the upper, lower, and
    overall means of the feature values.

    '''
    if dividers is None:
        m = mean(fv)
        lo = [v for v in fv if     v <= m]
        hi = [v for v in fv if m < v     ]
        dividers = [mean(lo) if lo else m, m, mean(hi) if hi else m]
    return distribution(fv, randomvariable(dividers), len(dividers) + 1)
