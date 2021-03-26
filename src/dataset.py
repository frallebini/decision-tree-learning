"""
Main source: https://github.com/aimacode/aima-python/blob/master/learning.py (as of late 2019)
"""

from utils import mean_boolean_error, parse_csv, remove_all, unique


class DataSet:
    """A data set for a machine learning problem. It has the following fields:
    d.examples   A list of examples. Each one is a list of attribute values.
    d.attrs      A list of integers to index into an example, so example[attr]
                 gives a value. Normally the same as range(len(d.examples[0])).
    d.attr_names Optional list of mnemonic names for corresponding attrs.
    d.target     The attribute that a learning algorithm will try to predict.
                 By default the final attribute.
    d.inputs     The list of attrs without the target.
    d.values     A list of lists: each sublist is the set of possible
                 values for the corresponding attribute. If initially None,
                 it is computed from the known examples by self.set_problem.
                 If not None, an erroneous value raises ValueError.
    d.distance   A function from a pair of examples to a non-negative number.
                 Should be symmetric, etc. Defaults to mean_boolean_error
                 since that can handle any field types.
    d.name       Name of the data set (for output display only).
    d.source     URL or other source where the data came from.
    d.exclude    A list of attribute indexes to exclude from d.inputs. Elements
                 of this list can either be integers (attrs) or attr_names.
    Normally, you call the constructor and you're done; then you just
    access fields like d.examples and d.target and d.inputs."""

    def __init__(self, examples=None, attrs=None, attr_names=None, target=-1, inputs=None,
                 values=None, distance=mean_boolean_error, name='', source='', exclude=()):
        """Accepts any of DataSet's fields. Examples can also be a
        string or file from which to parse examples using parse_csv.
        Optional parameter: exclude, as documented in .set_problem().
        >>> DataSet(examples='1, 2, 3')
        <DataSet(): 1 examples, 3 attributes>"""
        self.name = name
        self.source = source
        self.values = values
        self.distance = distance
        self.got_values_flag = bool(values)

        # initialize .examples from string, file, or list
        if isinstance(examples, str):
            self.examples = parse_csv(examples)
        else:
            try:
                self.examples = parse_csv(open('datasets/' + name + '.csv').read())
            except FileNotFoundError:
                self.examples = examples

        # attrs are the indices of examples, unless otherwise stated.
        if self.examples and not attrs:
            attrs = list(range(len(self.examples[0])))

        self.attrs = attrs

        # initialize .attr_names from string, file, or by default
        if isinstance(attr_names, str):
            self.attr_names = attr_names.split()
        else:
            self.attr_names = attr_names or attrs

        self.set_problem(target, inputs=inputs, exclude=exclude)

    def set_problem(self, target, inputs=None, exclude=()):
        """Set (or change) the target and/or inputs.
        This way, one DataSet can be used multiple ways. inputs, if specified,
        is a list of attributes, or specify exclude as a list of attributes
        to not use in inputs. Attributes can be -n .. n, or an attr_name.
        Also computes the list of possible values, if that wasn't done yet."""
        self.target = self.attr_num(target)
        exclude = list(map(self.attr_num, exclude))
        if inputs:
            self.inputs = remove_all(self.target, inputs)
        elif self.attrs:
            self.inputs = [a for a in self.attrs if a != self.target and a not in exclude]
        if not self.values:
            self.update_values()
        self.check_me()

    def check_me(self):
        """Check that my fields make sense."""
        assert len(self.attr_names) == len(self.values)
        if self.examples:
            assert len(self.attr_names) == len(self.attrs)
            assert self.target in self.attrs
            assert self.target not in self.inputs
            assert set(self.inputs).issubset(set(self.attrs))
            if self.got_values_flag:
                # only check if values are provided while initializing DataSet
                list(map(self.check_example, self.examples))

    def add_example(self, example):
        """Add an example to the list of examples, checking it first."""
        self.check_example(example)
        self.examples.append(example)

    def check_example(self, example):
        """Raise ValueError if example has any invalid values."""
        if self.values:
            for a in self.attrs:
                if example[a] not in self.values[a]:
                    raise ValueError('Bad value {} for attribute {} in {}'
                                     .format(example[a], self.attr_names[a], example))

    def attr_num(self, attr):
        """Returns the number used for attr, which can be a name, or -n .. n-1."""
        if isinstance(attr, str):
            return self.attr_names.index(attr)
        elif attr < 0:
            return len(self.attrs) + attr
        else:
            return attr

    def update_values(self):
        if self.examples:
            self.values = list(map(unique, zip(*self.examples)))

    def __repr__(self):
        return '<DataSet({}): {} examples, {} attributes>'.format(self.name, len(self.examples), len(self.attrs))
