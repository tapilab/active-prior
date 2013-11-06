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


def class_counts(y, n):
    """
    >>> class_counts([2, 1, 1, 0, 1, 2], 3)
    [1, 3, 2]
    """
    return [len([yj for yj in y if yj == yi]) for yi in range(n)]


def dist_to_labeled(xi, X):
    xid = xi.toarray()[0]
    return np.mean([spatial.distance.cosine(xid, xj.toarray()[0]) for xj in X])


def class_distr_match(pred, truth, n):
    """
    >>> class_distr_match([0, 1, 2], [2, 1, 0], 3)
    1.0
    >>> class_distr_match([0, 0, 0], [1, 1, 1], 2)
    0.0
    >>> class_distr_match([0, 0, 1, 1], [1, 1, 1, 1], 2)
    0.5
    """
    p_cc = 1. * np.array(class_counts(pred, n)) / len(pred)
    t_cc = 1. * np.array(class_counts(truth, n)) / len(truth)
    s = 1.0 - np.mean(np.abs(p_cc - t_cc))
    # cosine has weird property: cos([0, .5], [0, 1]) = 1 !
    # s = 1. - spatial.distance.cosine(p_cc, t_cc)
    return s
    # print 'pred=', p_cc, 'truth=', t_cc, 's=', s


class ActiveLearner(object):
    """ Abstract class for an active learning classifier. """

    __metaclass__ = abc.ABCMeta

    def __init__(self, clf, batch_size=1, iters=100, eval_f='accuracy_score'):
        self.clf = clf
        self.batch_size = batch_size
        self.iters = iters
        self.eval_f = getattr(sklearn.metrics, eval_f)

    def setup(self, xtrain, ytrain, xtest, ytest, labeled):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.ytest = ytest
        self.n_classes = len(set(ytest))

    def run(self, xtrain, ytrain, xtest, ytest, labeled):
        """ Fit the classifier using active learning. """
        self.setup(xtrain, ytrain, xtest, ytest, labeled)
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


class Combo(ActiveLearner):
    """ Alternate among a list of ActiveLearners. """

    def __init__(self, *args, **kwargs):
        self.learners = kwargs.pop('learners')
        super(Combo, self).__init__(*args, **kwargs)
        self.which = 0

    def setup(self, xtrain, ytrain, xtest, ytest, labeled):
        """ Fit the classifier using active learning. """
        for l in self.learners:
            l.setup(xtrain, ytrain, xtest, ytest, labeled)
        super(Combo, self).setup(xtrain, ytrain, xtest, ytest, labeled)

    def select_instances(self, X, labeled_indices, unlabeled_indices):
        learner = self.learners[self.which]
        self.which = (self.which + 1) % len(self.learners)
        return learner.select_instances(X, labeled_indices, unlabeled_indices)


class CertainUncertain(Combo):
    """ Select instances by alternating between certainty score and
    uncertainty score. """

    def __init__(self, *args, **kwargs):
        kwargs['learners'] = [Certain(*args, **kwargs), Uncertain(*args, **kwargs)]
        super(CertainUncertain, self).__init__(*args, **kwargs)


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

    def score_instance(self, idx, X, labeled_indices, unlabeled_indices):
        labeled_tmp = set(labeled_indices)
        unlabeled_tmp = set(unlabeled_indices)
            # add to training
        labeled_tmp |= set([idx])
        unlabeled_tmp -= set([idx])
            # train
        self.clf.fit(self.xtrain[list(labeled_tmp)], self.ytrain[list(labeled_tmp)])
            # predict on testing
        return class_distr_match(self.clf.predict(self.xtest), self.ytest, self.n_classes)

    def select_instances(self, X, labeled_indices, unlabeled_indices):
        scores = []
        unl = np.array(list(unlabeled_indices))
        for idx in unl:
            scores.append(self.score_instance(idx, X, labeled_indices, unlabeled_indices))
        print 'top scores=', sorted(scores, reverse=True)[:10]
        top_indices = np.array(scores).argsort()[::-1][:self.batch_size]
        # top_indices = np.array(scores).argsort()[:self.batch_size]
        self.clf.fit(self.xtrain[list(labeled_indices)], self.ytrain[list(labeled_indices)])
        topx = self.xtrain[unl[top_indices[0]]]
        print '\nscore=', scores[top_indices[0]], 'predictions=', self.clf.predict_proba(topx), 'truth=', self.ytrain[unl[top_indices[0]]], '#words=', topx.sum(), 'distance=', dist_to_labeled(topx, self.xtrain[list(labeled_indices)]), 'acc=', scores[top_indices[0]]
        return set(unl[top_indices])


class ClassDistrUncertCombo(Combo):
    """ Select instances by alternating between ClassDistrMatcherOracle and
    uncertainty score. """

    def __init__(self, *args, **kwargs):
        kwargs['learners'] = [ClassDistrMatcherOracle(*args, **kwargs), Uncertain(*args, **kwargs)]
        super(ClassDistrUncertCombo, self).__init__(*args, **kwargs)


class ClassDistrMatcherOracleUnc(ActiveLearner):
    """ Hybridizes ClassDistrMatcherOracle by first selecting the top K least
    certain instances, then ranking by class distribution match. """

    def __init__(self, *args, **kwargs):
        unc_size = kwargs.pop('unc_size', 50)
        super(ClassDistrMatcherOracleUnc, self).__init__(*args, **kwargs)
        self.uncertain = Uncertain(*args, **kwargs)
        self.uncertain.batch_size = unc_size

    def select_instances(self, X, labeled_indices, unlabeled_indices):
        unc_selected = self.uncertain.select_instances(X, labeled_indices, unlabeled_indices)
        scores = []
        unl = np.array(list(unc_selected))
        for idx in unl:
            labeled_tmp = set(labeled_indices)
            unlabeled_tmp = set(unlabeled_indices)
            # add to training
            labeled_tmp |= set([idx])
            unlabeled_tmp -= set([idx])
            # train
            self.clf.fit(self.xtrain[list(labeled_tmp)], self.ytrain[list(labeled_tmp)])
            # predict on testing
            scores.append(class_distr_match(self.clf.predict(self.xtest), self.ytest, self.n_classes))
        print 'top scores=', sorted(scores, reverse=True)[:10]
        top_indices = np.array(scores).argsort()[::-1][:self.batch_size]
        self.clf.fit(self.xtrain[list(labeled_indices)], self.ytrain[list(labeled_indices)])
        topx = self.xtrain[unl[top_indices[0]]]
        print '\npredictions=', self.clf.predict_proba(topx), 'truth=', self.ytrain[unl[top_indices[0]]], '#words=', topx.sum(), 'distance=', dist_to_labeled(topx, self.xtrain[list(labeled_indices)]), 'acc=', scores[top_indices[0]]
        return set(unl[top_indices])
