""" Perform an active learning experiment. """

import argparse
import logging
import pickle

from sklearn.linear_model import LogisticRegression

import active


logger = logging.getLogger(__name__)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch-size', default=1, type=int, help='number of labels per iteration')
    ap.add_argument('--data', default='/data/active-prior/news.pkl', help='pickled data file')
    ap.add_argument('--eval', default='accuracy_score', help='sklearn.metrics evaluation function name')
    ap.add_argument('--iters', default=100, type=int, help='number of learning iterations')
    ap.add_argument('--model', default='Random', help='name of ActiveLearner subclass')
    return ap.parse_args()


def construct_learner(args, clf):
    return getattr(active, args.model)(clf, batch_size=args.batch_size, iters=args.iters, eval_f=args.eval)


def main():
    args = parse_args()
    clf = LogisticRegression()
    learner = construct_learner(args, clf)
    print args
    print learner
    data = pickle.load(open(args.data, 'rb'))
    results = learner.run(data.xtrain, data.ytrain, data.xtest, data.ytest, set(range(10)))
    logging.info(results)


if __name__ == '__main__':
    main()
