# Naive Bayes Classifiers

These classifiers were part of a project for my Fall 2011 [machine learning class](http://www.ccs.neu.edu/home/jaa/CS6140.11F/).

To run, put *spambase.data* from the UCI [spambase dataset](http://archive.ics.uci.edu/ml/datasets/Spambase) into the same directory as *main.py*.

    $ python2 main.py

This will perform a 10-fold cross-validation analysis once for each of the feature models: bernoulli, gaussian, and histogram. Each will print a table describing the algorithm's performance on each fold, and a write a file containing data for a ROC curve.
