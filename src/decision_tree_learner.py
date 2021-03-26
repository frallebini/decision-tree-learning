"""
Main sources: http://aima.cs.berkeley.edu/python/learning.html
              https://github.com/aimacode/aima-python/blob/master/learning.py (as of late 2019)
"""

from learner import Learner
from decision_tree import DecisionTree
from utils import remove_all, argmax_random_tie, normalize
from math import log2


class DecisionTreeLearner(Learner):

    def train(self, dataset):
        self.dataset = dataset
        self.target = self.dataset.target
        self.values = self.dataset.values
        self.attr_names = self.dataset.attr_names
        self.tree = self.decision_tree_learning(dataset.examples, dataset.inputs)

    def predict(self, example):
        if isinstance(self.tree, DecisionTree):
            return self.tree.predict(example)
        else:
            return self.tree

    def decision_tree_learning(self, examples, attrs, parent_examples=()):
        if len(examples) == 0:
            return self.plurality_value(parent_examples)
        elif self.all_same_class(examples):
            return examples[0][self.target]
        elif len(attrs) == 0:
            return self.plurality_value(examples)
        else:
            best = self.choose_attribute(attrs, examples)
            tree = DecisionTree(best, self.attr_names[best])
            for v, exs in self.split_by(best, examples):
                subtree = self.decision_tree_learning(exs, remove_all(best, attrs), examples)
                tree.add(v, subtree)
            return tree

    def plurality_value(self, examples):
        """Return the most popular target value for this set of examples."""
        return argmax_random_tie(self.values[self.target], key=lambda v: self.count(self.target, v, examples))

    def count(self, attr, val, examples):
        """Count the number of examples that have example[attr] = val."""
        return sum(e[attr] == val for e in examples)

    def all_same_class(self, examples):
        """Are all these examples in the same target class?"""
        class0 = examples[0][self.target]
        return all(e[self.target] == class0 for e in examples)

    def choose_attribute(self, attrs, examples):
        """Choose the attribute with the highest information gain."""
        return argmax_random_tie(attrs, key=lambda a: self.information_gain(a, examples))

    def information_gain(self, attr, examples):
        """Return the expected reduction in entropy from splitting by attr."""

        def entropy(examples):
            target_values_count = [self.count(self.target, v, examples) for v in self.values[self.target]]
            probabilities = normalize(remove_all(0, target_values_count))
            return sum(-p * log2(p) for p in probabilities)

        n = len(examples)
        remainder = sum((len(exs) / n) * entropy(exs) for _, exs in self.split_by(attr, examples))
        return entropy(examples) - remainder

    def split_by(self, attr, examples):
        """Return a list of (val, examples) pairs for each val of attr."""
        return [(v, [e for e in examples if e[attr] == v]) for v in self.values[attr]]
