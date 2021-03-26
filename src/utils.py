"""
Main sources: https://github.com/aimacode/aima-python/blob/master/utils.py (as of late 2019)
              https://github.com/aimacode/aima-python/blob/master/learning.py (as of late 2019)
"""

from random import shuffle
from statistics import mean


def remove_all(item, seq):
    """Return a copy of seq with all occurrences of item removed."""
    return [x for x in seq if x != item]


def unique(seq):
    """Remove duplicate elements from seq. Assumes hashable elements."""
    return list(set(seq))


def remove_duplicates(seq):
    """Same as unique(), but for non-hashable elements."""
    new_seq = []
    for x in seq:
        if x not in new_seq:
            new_seq.append(x)
    return new_seq


def argmax_random_tie(seq, key=lambda x: x):
    """Return an element with highest fn(seq[i]) score; break ties at random."""
    return max(shuffled(seq), key=key)


def shuffled(iterable):
    """Randomly shuffle a copy of iterable."""
    items = list(iterable)
    shuffle(items)
    return items


def num_or_str(x):
    """The argument is a string; convert to a number if possible, or strip it."""
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return str(x).strip()


def normalize(dist):
    """Multiply each number by a constant such that the sum is 1.0"""
    total = sum(dist)
    return [(n / total) for n in dist]


def mean_boolean_error(x, y):
    return mean(_x != _y for _x, _y in zip(x, y))


def parse_csv(input_str, delim=','):
    r"""
    Input is a string consisting of lines, each line has comma-delimited
    fields. Convert this into a list of lists. Blank lines are skipped.
    Fields that look like numbers are converted to numbers.
    The delim defaults to ',' but '\t' and None are also reasonable values.
    >>> parse_csv('1, 2, 3 \n 0, 2, na')
    [[1, 2, 3], [0, 2, 'na']]
    """
    lines = [line for line in input_str.splitlines() if line.strip()]
    return [list(map(num_or_str, line.split(delim))) for line in lines]


def print_time(time):
    """Prints a number representing a time in seconds and the equivalent in
    minutes if time >= 60sec."""
    mins = int(time/60)
    if mins > 0:
        print('Execution time = {:.3f} seconds (~{} minutes)'.format(time, mins))
    else:
        print('Execution time = {:.3f} seconds'.format(time))
