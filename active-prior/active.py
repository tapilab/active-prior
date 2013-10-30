"""
Active Learning classes.
"""
import abc
import logging
import math
import random
import sys

import numpy as np
import sklearn.metrics


logger = logging.getLogger(__name__)


def entropy(probas):
    """
    Negative of sum of p * lg (p)
    >>> entropy([.5, .5])
    1.0
    >>> entropy([.5, .5, .5])
    1.5
    >>> entropy([1., 0.])
    -0.0
    >>> entropy([.5, .5, 0.])
    1.0
    >>> entropy([.25, .25, .25, .25])
    2.0
    """
    return -sum(p * math.log(p, 2.) for p in probas if p != 0.)


class ActiveLearner(object):
    """ Abstract class for an active learning classifier. """

    __metaclass__ = abc.ABCMeta

    def __init__(self, clf, batch_size=1, iters=100, eval_f='accuracy_score'):
        self.clf = clf
        self.batch_size = batch_size
        self.iters = iters
        self.eval_f = getattr(sklearn.metrics, eval_f)

    def run(self, xtrain, ytrain, xtest, ytest, labeled):
        """ Fit the classifier using active learning. """
        self.clf.fit(xtrain[list(labeled)], ytrain[list(labeled)])
        unlabeled = set(range(len(ytrain))) - labeled
        results = []
        for iteri in range(self.iters):
            # logger.info('\riteration %d', iteri)
            sys.stdout.write('\riteration %d' % iteri)
            sys.stdout.flush()
            tolabel = self.select_instances(xtrain, labeled, unlabeled)
            # logging.debug('labeling %s', str(tolabel))
            labeled |= tolabel
            unlabeled -= tolabel
            self.clf.fit(xtrain[list(labeled)], ytrain[list(labeled)])
            results.append(self.eval_f(self.clf.predict(xtest), ytest))
        return results

    @abc.abstractmethod
    def select_instances(self, X, labeled_indices, unlabeled_indices):
        """ Return the set of indices from X to label. """
        return


class Random(ActiveLearner):
    """ Select instances at random. """

    def __init__(self, *args, **kwargs):
        super(Random, self).__init__(*args, **kwargs)

    def select_instances(self, X, labeled_indices, unlabeled_indices):
        a = list(unlabeled_indices)
        random.shuffle(a)
        return set(a[:self.batch_size])


class Uncertain(ActiveLearner):
    """ Select instances by uncertainty score. """

    def __init__(self, *args, **kwargs):
        super(Uncertain, self).__init__(*args, **kwargs)

    def select_instances(self, X, labeled_indices, unlabeled_indices):
        unl = np.array(list(unlabeled_indices))
        top_indices = self.clf.predict_proba(X[unl]).max(axis=1).argsort()[:self.batch_size]
        return set(unl[top_indices])


class Certain(ActiveLearner):
    """ Select instances by certainty score. """

    def __init__(self, *args, **kwargs):
        super(Certain, self).__init__(*args, **kwargs)

    def select_instances(self, X, labeled_indices, unlabeled_indices):
        unl = np.array(list(unlabeled_indices))
        top_indices = self.clf.predict_proba(X[unl]).max(axis=1).argsort()[::-1][:self.batch_size]
        return set(unl[top_indices])
