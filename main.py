# stdlib
import math
import collections
# local
import progress as pg
import gaussian
import bernoulli
import histogram


def load(s, f):
    '''Load a single feature value from a string.'''
    changetype, checkvalue = f
    d = changetype(s)
    if checkvalue(d):
        return d
    else:
        raise ValueError(d + ' fails value check in ' + f)


def pullone(sequence):
    '''Each iteration yield one entry and the rest of the sequence.'''
    for i, s in enumerate(sequence):
        yield (s, sequence[:i] + sequence[i + 1:])


def naivebayes(testing, training, model):
    '''Expect two lists of labeled datapoints and a model.

    A data point is a dictionary with
        'features' : list<number>
        'label' : boolean

    A model is a function which takes a feature vector and produces either a
    probability mass function or a probability density function. The function
    produced takes a value and calculates a score which can be used to compare
    relative likelyhoods.
    [list<number> --> [number --> number]]

    A feature vector is all the values of one feature accross all data points.
    list<number>

    Return a copy of testing in which the datapoints have no features and are
    augmented with a real-valued 'logodds' field.

    '''
    # split by label: list<datapoint>
    T = []
    F = []
    for dp in training:
        (T if dp['label'] else F).append(dp['features'])
    # assess prior log-odds belief: number
    Tprior = float(len(T)) / len(training)
    Fprior = float(len(F)) / len(training)
    priorlogodds = math.log(Tprior / Fprior)
    # generate models of each feature: list<function>
    # zip(*seq) rotates data points to feature vectors
    Tm = []
    Fm = []
    for Tfv, Ffv in zip(zip(*T), zip(*F)):
        try:
            tm = model(Tfv)
            fm = model(Ffv)
        except gaussian.ZeroVarianceError:
            # feature has a class-conditional variance of zero for T or F
            # ignore this feature with math.log(tm=1.0 / tm=1.0) == 0.0
            tm = fm = lambda x: 1.0
        Tm.append(tm)
        Fm.append(fm)
    # copy testing: list<datapoint>
    ret = [dp.copy() for dp in testing]
    # accumulate log-odds over the features of each data point: list<datapoint>
    zct = len(T[0]) * [0] # a zero for each feature
    for dp in ret:
        dp['logodds'] = priorlogodds
        for idx, val in enumerate(dp['features']):
            gT = Tm[idx](val)
            gF = Fm[idx](val)
            if gT != 0.0 and gF != 0.0:
                dp['logodds'] += math.log(gT / gF)
            else:
                zct[idx] += 1
        del dp['features']
    # notify about features with ignored model results
    if any(zct):
        print 'naivebayes(): Ignored features: {}'.format\
              ('; '.join(['#{}, {}x'.format(i, ct) for i, ct in enumerate(zct) if ct > 0.01 * len(testing)]))
    # return the augmented copy of training
    return ret


def minerrop(augmented):
    '''Return the operating point which minimizes overall error.'''
    op = 0.0
    res = analyze(applyop(op, augmented))
    direction = 0.1 * (1 if res['fpr'] > res['fnr'] else -1)
    preverr = res['oer']
    while True:
        op += direction
        res = analyze(applyop(op, augmented))
        if res['oer'] > preverr:
            return op - direction
        else:
            preverr = res['oer']


def applyop(op, augmented):
    '''Assign a prediction label to each augmented datapoint.'''
    for dp in augmented:
        dp['prediction'] = int(dp['logodds'] > op)
    return augmented


def analyze(predicted):
    '''Produce the confusion matrix and error-tables data.

    Takes a list of datapoints with or without features, augmented with
    log-odds and a label prediction.

    '''
    results = {'tp':0.0, 'fn':0.0, 'fp':0.0, 'tn':0.0}
    for dp in predicted:
        results['tp'] += dp['label'] == 1 and dp['prediction'] == 1
        results['fn'] += dp['label'] == 1 and dp['prediction'] == 0
        results['fp'] += dp['label'] == 0 and dp['prediction'] == 1
        results['tn'] += dp['label'] == 0 and dp['prediction'] == 0

    # false positive rate (false alarms)
    # fraction of negatives which are misclassified as positive
    # fraction of ALL legit emails which are in the spam folder
    results['fpr'] = results['fp'] / (results['fp'] + results['tn'])

    # false negative rate (missed spams)
    # fraction of positives which are misclassified as negative
    # fraction of the ALL spam emails which are in the inbox
    results['fnr'] = results['fn'] / (results['tp'] + results['fn'])

    # true positive rate (detections)
    # fraction of positives which are classified as positive
    # fraction of the ALL spam emails which are in the spam box
    results['tpr'] = results['tp'] / (results['fn'] + results['tp'])

    # overall error rate (mistakes)
    # fraction of all data points which are misclassified
    # fraction of ALL emails which are someplace they aren't supposed to be
    results['oer'] = (results['fp'] + results['fn']) / len(predicted)
    #
    return results


def rocdata(results):
    '''Output all (tp rate, fp rate) pairs accross operating points.'''
    # sort data points by logodds
    results['dpresults'].sort(key=lambda x: x['logodds'])
    # make first pair s.t. for all dp, prediction is True
    r = analyze(applyop(results['dpresults'][0]['logodds'] - 1, results['dpresults'][:]))
    pairs = [(r['fpr'], r['tpr'])]
    # make the rest s.t. on the final one, for all dp prediction is False
    for dp in results['dpresults']:
        r = analyze(applyop(dp['logodds'], results['dpresults'][:]))
        pairs.append((r['fpr'], r['tpr']))
    return pairs


if __name__ == '__main__':
    #
    DATAFILE = 'spambase.data'
    #
    DATAFORMAT = 48 * [(float, lambda x: 0 <= x <= 100)] +\
                  6 * [(float, lambda x: 0 <= x <= 100)] +\
                  1 * [(float, lambda x: 0 <= x)] +\
                  1 * [(int, lambda x: 0 <= x)] +\
                  1 * [(int, lambda x: 0 <= x)] +\
                  1 * [(int, lambda x: x == 1 or x == 0)]
    #
    TABLE_ = '+------------+------------+------------+------------+------------+------------+------------+'
    TABLEH = '| Test Fold  | Test Ct    | Train Ct   | Operat Pnt | FP Rate    | FN Rate    | Error Rate |'
    TABLE  = '| {: ^10} | {: >10} | {: >10} | {: <10.4f} | {: <10.8f} | {: <10.8f} | {: <10.8f} |'
    #
    FOLDCOUNT = 10
    ROCFOLD = 0
    folds = [[] for i in xrange(FOLDCOUNT)]
    k = 0 # kurrent fold
    #
    print
    print 'Loading "{}"'.format(DATAFILE)
    with open(DATAFILE, mode='rb') as fd:
        with pg.Progress(4601, timeout=2, callback=pg.bar('Loading', 32)) as pr:
            for line in fd:
                # clean the data & create the datapoint
                cleandata = [load(raw.strip(), fmt) for raw, fmt in zip(line.split(','), DATAFORMAT)]
                datapoint = {'features':cleandata[:-1], 'label':cleandata[-1]}
                # add to the current fold & switch to the next fold
                folds[k].append(datapoint)
                k = (k + 1) % FOLDCOUNT
                # indicate progress
                pr.next()
    #
    results = collections.defaultdict(list)
    for name, model in [(m.__name__, m.model) for m in [bernoulli, gaussian, histogram]]:
        print
        print '10-fold cross-validation for {} Naive Bayes'.format(name.capitalize())
        with pg.Progress(FOLDCOUNT, timeout=4, callback=pg.bar(name.capitalize(), 32)) as pr:
            for k, (testing, training) in enumerate(pullone(folds)):
                # flatten training folds from list<list<datapoint>> to list<datapoint>
                training = reduce(lambda acc, cur: acc + cur, training)
                # calculate log-odds for the current model
                augmented = naivebayes(testing, training, model)
                # find a good operating point
                op = minerrop(augmented)
                # assign predictions
                augmented = applyop(op, augmented)
                # analyze and keep the results around
                r = analyze(augmented)
                r['operatingpoint'] = op
                r['trainingsize'] = len(training)
                r['testingfold'] = k
                r['dpresults'] = augmented
                results[name].append(r)
                # indicate progress
                pr.next()
        print TABLE_
        print TABLEH
        print TABLE_
        for r in results[name]:
            print TABLE.format(r['testingfold'], len(r['dpresults']), r['trainingsize'],
                               r['operatingpoint'], r['fpr'], r['fnr'], r['oer'])
        print TABLE_
        print TABLE.format('Average', '', '',
                           sum([r['operatingpoint'] for r in results[name]]) / FOLDCOUNT,
                           sum([r['fpr'] for r in results[name]]) / FOLDCOUNT,
                           sum([r['fnr'] for r in results[name]]) / FOLDCOUNT,
                           sum([r['oer'] for r in results[name]]) / FOLDCOUNT)
        print TABLE_
    #
    for name, r in [(k, v[ROCFOLD]) for k, v in results.iteritems()]:
        roc = rocdata(r)
        with open('predmond-rocdata-{}-fold{}.txt'.format(name, ROCFOLD), mode='wb') as fd:
            [(fd.write('{}, {}'.format(*p)), fd.write('\n')) for p in roc]
