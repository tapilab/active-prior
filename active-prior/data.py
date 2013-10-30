""" Training/testing data. """


class Data(object):

    def __init__(self, xtrain, ytrain, xtest, ytest):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.ytest = ytest
