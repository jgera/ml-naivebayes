# Naive Bayes Classifiers

These classifiers were part of a [project](http://www.ccs.neu.edu/home/jaa/CS6140.11F/Homeworks/hw.02.html) for my Fall 2011 [machine learning class](http://www.ccs.neu.edu/home/jaa/CS6140.11F/).

To run, put *spambase.data* from the UCI [spambase dataset](http://archive.ics.uci.edu/ml/datasets/Spambase) into the same directory as *main.py*.

    $ python2 main.py

This will perform a 10-fold cross-validation analysis once for each of the feature models: bernoulli, gaussian, and histogram. Each will print a table describing its performance on each fold and a write a file containing data for a ROC curve.

See [my analysis](https://docs.google.com/document/d/1ES3X8PE1vNi_l_5n0jVYY8psTdbKZQv3T4dssZU6sko/edit) for a discussion of the results.

-- [PLR](http://f06mote.com)
