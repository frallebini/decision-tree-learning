"""
A test that performs 10-fold cross-validation on the chosen data set with and
without rule post-pruning using the chosen number of examples. The decision
tree (w/ rule post-pruning) and the set of rules (w/o rule post-pruning)
produced during the last iteration of cross_validation() are also printed on
screen.
"""

from testing_functions import cross_validation_time, choose_dataset
from decision_tree_learner import DecisionTreeLearner
from rule_post_pruning_learner import RulePostPruningLearner
from utils import print_time

dataset = choose_dataset()

title = '10-FOLD CROSS-VALIDATION on ' + dataset.name + ' data set'
print('\n' + '=' * len(title))
print(title)
print('=' * len(title))

print('\n~~~~~~~~~~~~~~~~~~~~~~~~~')
print('WITHOUT RULE POST-PRUNING')
print('~~~~~~~~~~~~~~~~~~~~~~~~~\n')
learner = DecisionTreeLearner()
result, time = cross_validation_time(learner, dataset)
print('Proportion of correctly classified examples = {:.3f}'.format(result))
print_time(time)
print('Last tree = ')
learner.tree.display()

print('\n~~~~~~~~~~~~~~~~~~~~~~')
print('WITH RULE POST-PRUNING')
print('~~~~~~~~~~~~~~~~~~~~~~\n')
learner = RulePostPruningLearner()
result, time = cross_validation_time(learner, dataset)
print('Proportion of correctly classified examples = {:.3f}'.format(result))
print_time(time)
print('Last set of rules = ')
learner.set_of_rules.display()
