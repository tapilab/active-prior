"""
Active Learning classes.
"""
import abc
import logging
import random

import sklearn.metrics


logger = logging.getLogger(__name__)


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
        unlabeled = set(range(len(ytrain))) - labeled
        results = []
        for iteri in range(self.iters):
            logger.info('iteration %d', iteri)
            tolabel = self.select_instances(xtrain, labeled, unlabeled)
            logging.debug('labeling %s', str(tolabel))
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

    def __init__(self, clf, batch_size=1, iters=100, eval_f='accuracy_score'):
        super(Random, self).__init__(clf, batch_size, iters, eval_f)

    def select_instances(self, X, labeled_indices, unlabeled_indices):
        a = list(unlabeled_indices)
        random.shuffle(a)
        return set(a[:self.batch_size])
