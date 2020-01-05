"""
Source: http://aima.cs.berkeley.edu/python/learning.html
"""


class Learner:
    """A Learner, or Learning Algorithm, can be trained with a data set,
    and then asked to predict the target attribute of an example."""

    def train(self, dataset):
        raise NotImplementedError

    def predict(self, example):
        raise NotImplementedError
