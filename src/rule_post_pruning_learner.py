from decision_tree_learner import DecisionTreeLearner
from set_of_rules import SetOfRules
from testing_functions import test
from utils import remove_all, remove_duplicates
from random import shuffle


class RulePostPruningLearner(DecisionTreeLearner):

    def train(self, dataset):
        """Uses a third of dataset examples for training and the rest for
        validation. Once it has been trained, it holds a SetOfRules obtained by
        converting into rules the DecisionTree produced by a
        DecisionTreeLearner trained on the same training examples. The rules
        are then pruned according to their accuracy on the validation examples."""
        examples = dataset.examples

        total_size = len(examples)
        validation_size = total_size // 3
        training_size = total_size - validation_size

        dataset.examples = examples[:training_size]
        self.validation_examples = examples[training_size:total_size]
        super().train(dataset)

        self.set_of_rules = SetOfRules(dataset, self.tree)
        self.input_names = remove_all(self.attr_names[self.target], self.attr_names)

        self.set_of_rules.rules = remove_duplicates([self.prune(rule) for rule in self.set_of_rules.rules])

        dataset.examples = examples

    def prune(self, rule):
        """Selects the precondition whose pruning (removal) produces the
        greatest increase in SetOfRules accuracy on the validation examples,
        and then removes it. Repeats this process on the remaining
        preconditions until further pruning decreases the accuracy or until
        there is only one precondition left."""
        if len(rule) <= 2:
            return rule
        pruned_name = self.choose_attr_name(rule)
        if pruned_name:
            del rule[pruned_name]
            return self.prune(rule)
        else:
            return rule

    def choose_attr_name(self, rule):
        """Returns the attribute name in the current precondition whose pruning
        causes the greatest increase in SetOfRules accuracy on the validation
        examples. If such increase is negative (i.e. it is a decrease), it
        means that any precondition, if pruned, decreases the accuracy, and
        therefore we must perform no pruning."""
        input_names = [n for n in self.input_names if n in rule.keys()]
        shuffle(input_names)
        scores = [self.accuracy_improvement(rule, n) for n in input_names]
        max_score = max(scores)
        if max_score >= 0:
            return input_names[scores.index(max_score)]
        else:
            return None

    def accuracy_improvement(self, rule, attr_name):
        """Computes the increase (or decrease) in SetOfRules accuracy on the
        validation examples obtained by pruning the precondition with
        attr_name."""
        old_score = test(self, self.dataset, self.validation_examples)
        attr_val = rule.pop(attr_name)
        new_score = test(self, self.dataset, self.validation_examples)
        rule[attr_name] = attr_val
        return new_score - old_score

    def predict(self, example):
        return self.set_of_rules.predict(example)
