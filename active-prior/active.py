"""
Active Learning classes.
"""
import abc
import logging
import math
import random
import sys

import numpy as np
from scipy import spatial
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


def class_counts(y):
    """
    >>> class_counts([2, 1, 1, 0, 1, 2])
    [1, 3, 2]
    """
    return [len([yj for yj in y if yj == yi]) for yi in sorted(set(y))]


def dist_to_labeled(xi, X):
    xid = xi.toarray()[0]
    return np.mean([spatial.distance.cosine(xid, xj.toarray()[0]) for xj in X])


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
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.ytest = ytest
        self.clf.fit(xtrain[list(labeled)], ytrain[list(labeled)])
        unlabeled = set(range(len(ytrain))) - labeled
        results = []
        for iteri in range(self.iters):
            sys.stdout.write('\riteration %d labeled=%d unlabeled=%d' % (iteri, len(labeled), len(unlabeled)))
            sys.stdout.flush()
            tolabel = self.select_instances(xtrain, labeled, unlabeled)
            labeled |= tolabel
            unlabeled -= tolabel
            self.clf.fit(xtrain[list(labeled)], ytrain[list(labeled)])
            results.append(self.eval_f(self.clf.predict(xtest), ytest))
        print '\n'
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


class CertainUncertain(ActiveLearner):
    """ Select instances by alternating between certainty score and
    uncertainty score. """

    def __init__(self, *args, **kwargs):
        super(CertainUncertain, self).__init__(*args, **kwargs)
        self.certain = Certain(*args, **kwargs)
        self.uncertain = Uncertain(*args, **kwargs)
        self.selectors = [self.certain, self.uncertain]
        self.which = 0

    def select_instances(self, X, labeled_indices, unlabeled_indices):
        selector = self.selectors[self.which]
        self.which = (self.which + 1) % len(self.selectors)
        return selector.select_instances(X, labeled_indices, unlabeled_indices)


class GreedyOracle(ActiveLearner):
    """ Select instance that maximizes accuracy on test set. This is
    cheating. """

    def __init__(self, *args, **kwargs):
        super(GreedyOracle, self).__init__(*args, **kwargs)

    def select_instances(self, X, labeled_indices, unlabeled_indices):
        scores = []
        unl = np.array(list(unlabeled_indices))
        for idx in unl:
            labeled_tmp = set(labeled_indices)
            unlabeled_tmp = set(unlabeled_indices)
            # add to training
            labeled_tmp |= set([idx])
            unlabeled_tmp -= set([idx])
            # train
            self.clf.fit(self.xtrain[list(labeled_tmp)], self.ytrain[list(labeled_tmp)])
            # predict on testing
            scores.append(self.eval_f(self.clf.predict(self.xtest), self.ytest))
        top_indices = np.array(scores).argsort()[::-1][:self.batch_size]
        self.clf.fit(self.xtrain[list(labeled_indices)], self.ytrain[list(labeled_indices)])
        topx = self.xtrain[unl[top_indices[0]]]
        print '\npredictions=', self.clf.predict_proba(topx), 'truth=', self.ytrain[unl[top_indices[0]]], '#words=', topx.sum(), 'distance=', dist_to_labeled(topx, self.xtrain[list(labeled_indices)]), 'acc=', scores[top_indices[0]]
        return set(unl[top_indices])


class ClassDistrMatcherOracle(ActiveLearner):
    """ Select instance such that the predicted class distribution matches that supplied. """

    def __init__(self, *args, **kwargs):
        super(ClassDistrMatcherOracle, self).__init__(*args, **kwargs)

    def class_distr_match(self, pred, truth):
        p_cc = 1. * np.array(class_counts(pred)) / len(pred)
        t_cc = 1. * np.array(class_counts(truth)) / len(pred)
        s = 1. - spatial.distance.cosine(p_cc, t_cc)
        return s

    def select_instances(self, X, labeled_indices, unlabeled_indices):
        scores = []
        unl = np.array(list(unlabeled_indices))
        for idx in unl:
            labeled_tmp = set(labeled_indices)
            unlabeled_tmp = set(unlabeled_indices)
            # add to training
            labeled_tmp |= set([idx])
            unlabeled_tmp -= set([idx])
            # train
            self.clf.fit(self.xtrain[list(labeled_tmp)], self.ytrain[list(labeled_tmp)])
            # predict on testing
            scores.append(self.class_distr_match(self.clf.predict(self.xtest), self.ytest))
        print 'top scores=', sorted(scores, reverse=True)[:10]
        top_indices = np.array(scores).argsort()[::-1][:self.batch_size]
        self.clf.fit(self.xtrain[list(labeled_indices)], self.ytrain[list(labeled_indices)])
        topx = self.xtrain[unl[top_indices[0]]]
        print '\npredictions=', self.clf.predict_proba(topx), 'truth=', self.ytrain[unl[top_indices[0]]], '#words=', topx.sum(), 'distance=', dist_to_labeled(topx, self.xtrain[list(labeled_indices)]), 'acc=', scores[top_indices[0]]
        return set(unl[top_indices])
