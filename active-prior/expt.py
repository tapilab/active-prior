""" Perform an active learning experiment. """

import argparse
from collections import defaultdict
import logging
import math
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


import active


logger = logging.getLogger(__name__)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch-size', default=1, type=int, help='number of labels per iteration')
    ap.add_argument('--data', default='/data/active-prior/news.pkl', help='pickled data file')
    ap.add_argument('--eval', default='roc_auc_score', help='sklearn.metrics evaluation function name')
    ap.add_argument('--init-labeled', default=20, type=int, help='initial number of labeled examples')
    ap.add_argument('--iters', default=100, type=int, help='number of learning iterations')
    ap.add_argument('--models', default='Random,Uncertain,Certain', help='name of ActiveLearner subclasses')
    ap.add_argument('--trials', default=10, type=int, help='number of random trials')
    return ap.parse_args()


def construct_learner(learner_name, clf, args):
    return getattr(active, learner_name)(clf, batch_size=args.batch_size, iters=args.iters, eval_f=args.eval)


def init_labeled(total, toselect):
    """ Sample indices of initially labeled examples. """
    initial_labeled = range(total)
    random.shuffle(initial_labeled)
    return set(initial_labeled[:toselect])


def plot_results(results, args):
    """ Plot final learning curves, with standard error. """
    colors = ['y', 'm', 'c', 'r', 'g', 'b']
    for learner, results in results.iteritems():
        x = range(len(results[0]))
        y = np.mean(results, axis=0)
        error = np.std(results, axis=0) / math.sqrt(len(results))
        color = colors.pop()
        plt.plot(x, y, color + 'o', label=learner)
        plt.fill_between(x, y - error, y + error, alpha=0.5, facecolor=color)
    plt.legend(loc='lower right')
    plt.xlabel('iteration')
    plt.ylabel(args.eval)
    plt.show()


def print_results(results):
    """ Print tab-separated table of results. """
    names = results.keys()
    vals = [np.mean(results[k], axis=0) for k in names]
    print '\t'.join('%10s' % n for n in names)
    for row in np.transpose(vals):
        print '\t'.join(['%.7g' % v for v in row])


def main():
    random.seed(1234567)
    args = parse_args()
    clf = LogisticRegression()
    # clf = MultinomialNB()
    data = pickle.load(open(args.data, 'rb'))
    results = defaultdict(lambda: [])
    for triali in range(args.trials):
        initial_labeled = init_labeled(len(data.ytrain), args.init_labeled)
        for learner_name in args.models.split(','):
            logger.info('trial %d for active learner %s' % (triali, learner_name))
            learner = construct_learner(learner_name, clf, args)
            # Note that we must clone initial_labeled each time, otherwise it is never cleared!
            res = learner.run(data.xtrain, data.ytrain, data.xtest,
                              data.ytest, set(initial_labeled))
            logging.info('\tmean=%g\n\n' % np.mean(res))
            results[learner_name].append(res)
    print_results(results)
    plot_results(results, args)

if __name__ == '__main__':
    main()
