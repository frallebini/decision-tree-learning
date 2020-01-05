"""
A test that plots the learning curve of the chosen data set using the chosen
number of examples. Each point is obtained by averaging over 20 trials.
"""

from decision_tree_learner import DecisionTreeLearner
from rule_post_pruning_learner import RulePostPruningLearner
from testing_functions import plot_learning_curve, choose_dataset
from utils import print_time

dataset = choose_dataset()
time = plot_learning_curve([DecisionTreeLearner(), RulePostPruningLearner()],
                           dataset,
                           'Learning curve for the ' + dataset.name + ' data set',
                           ['Without rule post-pruning', 'With rule post-pruning'])
print_time(time)
