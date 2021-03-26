"""
Main source: http://aima.cs.berkeley.edu/python/learning.html
"""


class DecisionTree:
    """A DecisionTree holds an attribute that is being tested, and a
    dict of {attr_val: Tree} entries.  If Tree here is not a DecisionTree
    then it is the final classification of the example."""

    def __init__(self, attr, attr_name=None, branches=None):
        """Initialize by saying what attribute this node tests."""
        self.attr = attr
        self.attr_name = attr_name or attr
        self.branches = branches or {}

    def predict(self, example):
        """Given an example, use the tree to classify the example."""
        child = self.branches[example[self.attr]]
        if isinstance(child, DecisionTree):
            return child.predict(example)
        else:
            return child

    def add(self, val, subtree):
        """Add branch {val: subtree}"""
        self.branches[val] = subtree

    def display(self, indent=0):
        name = self.attr_name
        print('Test', name)
        for val, subtree in self.branches.items():
            # items() does not guarantee any ordering in the tuples it returns
            print(' ' * 4 * indent, name, '=', val, '==>', end=' ')
            if isinstance(subtree, DecisionTree):
                subtree.display(indent + 1)
            else:
                print('RESULT =', subtree)

    def __repr__(self):
        return 'DecisionTree({}, {})'.format(self.attr_name, self.branches)
