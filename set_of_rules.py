from decision_tree_learner import DecisionTree
from os import remove
from utils import num_or_str


class SetOfRules:
    """A SetOfRules holds a list of rules of form preconditions + result
    corresponding to the paths from root to leaf of a DecisionTree. Each rule
    is a dictionary with {attr_name: attr_value} key–value pairs."""

    def __init__(self, dataset, tree):
        """Assumes that tree is the DecisionTree resulting from training a
        DecisionTreeLearner on DataSet dataset."""
        self.dataset = dataset
        self.attr_names = dataset.attr_names
        self.target_name = self.attr_names[dataset.target]
        self.inputs = dataset.inputs
        self.convert_to_rules(tree)

    def convert_to_rules(self, tree):
        """Builds the list of rules corresponding to the paths from root to
        leaf in tree. As an intermediate step, a file rules.txt is produced,
        containing those rules in a human-readable format; this file is then
        removed from the current directory once the list of rules has been
        created."""
        f = open('rules.txt', 'w')

        def write_paths(tree, path, attr_value=None):
            """Visits tree depth-first and writes down in rules.txt the paths
            from root to leaf."""
            if isinstance(tree, DecisionTree):
                if path:
                    path[-1] += ' = ' + str(attr_value)
                path.append(tree.attr_name)
                for value, subtree in tree.branches.items():
                    write_paths(subtree, path, value)
            else:
                if path:
                    path[-1] += ' = ' + str(attr_value)
                path.append(str(self.target_name) + ' = ' + str(tree))
                f.write(str(path) + '\n')
            path.pop()
            if path:
                path[-1] = path[-1].split()[0]

        write_paths(tree, [])
        f.close()

        # rules.txt cleanup
        with open('rules.txt') as f:
            rules_str = f.read().replace("'", '').replace('[', '').replace(']', '')
        remove('rules.txt')

        # text file to list of dictionaries conversion
        self.rules = []
        for rule_str in rules_str.splitlines():
            rule = {}
            for condition in rule_str.split(', '):
                attr_name, attr_val = list(map(num_or_str, condition.split('=')))
                rule[attr_name] = attr_val
            self.rules.append(rule)

    def predict(self, example):
        for rule in self.rules:
            if self.fits(rule, example):
                return rule[self.target_name]
        return None

    def fits(self, rule, example):
        """Returns True if rule has the same attribute values (target excluded)
        as example."""
        for i in self.inputs:
            for attr_name, attr_val in rule.items():
                if self.attr_names[i] == attr_name and example[i] != attr_val:
                    return False
        return True

    def display(self):
        dnf = ''  # dnf stands for Disjunctive Normal Form
        for rule in self.rules:
            conj = ''
            for attr_name, attr_val in rule.items():
                if conj:
                    conj += ' AND ' + str(attr_name) + ' = ' + str(attr_val)
                else:
                    conj += '(' + str(attr_name) + ' = ' + str(attr_val)
            dnf += conj + ') OR\n'
        dnf = dnf[:-4]
        print(dnf)

    def __repr__(self):
        return self.rules.__repr__()
