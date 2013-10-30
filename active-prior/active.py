"""
Active Learning classes.
"""
import abc
import logging

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

    @abc.abstractmethod
    def run(self, X, y):
        """ Fit the classifier using active learning. """
        return


class Random(ActiveLearner):
    """ Select instances at random. """

    def __init__(self, clf, batch_size=1, iters=100, eval_f='accuracy_score'):
        super(Random, self).__init__(clf, batch_size, iters, eval_f)

    def run(self, xtrain, ytrain, xtest, ytest):
        for iteri in range(self.iters):
            logger.info('iteration %d', iteri)
        return
