"""
Main source: http://aima.cs.berkeley.edu/python/learning.html
"""

from random import shuffle
from statistics import mean
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from dataset import DataSet
from restaurant_dataset import SyntheticRestaurant


def test(learner, dataset, examples=None, verbose=0):
    """Return the proportion of the examples that are correctly predicted.
    Assumes the learner has already been trained.
    verbose — 0: No output; 1: Output wrong; 2 (or greater): Output correct."""
    examples = examples or dataset.examples
    if len(examples) == 0:
        return 0.0
    right = 0
    for example in examples:
        desired = example[dataset.target]
        output = learner.predict(example)
        if output == desired:
            right += 1
            if verbose >= 2:
                print('OK: got {} for {}'.format(desired, example))
        elif verbose:
            print('WRONG: got {}, expected {} for {}'.format(output, desired, example))
    return right / len(examples)


def train_and_test(learner, dataset, start, end):
    """Reserve dataset.examples[start:end] for test; train on the remainder.
    Return the proportion of examples correct on the test examples."""
    examples = dataset.examples
    try:
        dataset.examples = examples[:start] + examples[end:]
        learner.train(dataset)
        return test(learner, dataset, examples[start:end])
    finally:  # executed after the return statement: go back to the original dataset
        dataset.examples = examples


def learning_curve(learner, dataset, trials=20, sizes=None):
    n = len(dataset.examples)
    if not sizes:
        sizes = range(1, n-2)

    def score(size):
        shuffle(dataset.examples)
        return train_and_test(learner, dataset, 0, size)

    proportion_correct = [mean(score(size) for _ in range(trials)) for size in sizes]
    return list(reversed([n - size for size in sizes])), list(reversed(proportion_correct))


def plot_learning_curve(learners, dataset, title, legend_labels, trials=20, sizes=None):
    """Calls learning_curve() on two learners, plots the results,
    and returns how much time it took."""
    start = timer()
    x, y1 = learning_curve(learners[0], dataset, trials, sizes)
    _, y2 = learning_curve(learners[1], dataset, trials, sizes)
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.xlabel('Training set size')
    plt.ylabel('Proportion correct on test set')
    plt.title(title)
    plt.legend(legend_labels, loc='lower right')
    plt.tight_layout()
    plt.show()
    end = timer()
    return end-start


def cross_validation(learner, dataset, k=10, trials=1):
    """Do k-fold cross_validate and return their mean.
    That is, keep out 1/k of the examples for testing on each of k runs.
    Shuffle the examples first; If trials>1, average over several shuffles."""
    if not k:
        k = len(dataset.examples)
    if trials > 1:
        return mean([cross_validation(learner, dataset, k, trials=1)
                     for _ in range(trials)])
    else:
        n = len(dataset.examples)
        shuffle(dataset.examples)
        return mean([train_and_test(learner, dataset, i*(n//k), (i+1)*(n//k))
                     for i in range(k)])


def cross_validation_time(learner, dataset, k=10, trials=1):
    """Calls cross_validation() and measures its execution time."""
    start = timer()
    result = cross_validation(learner, dataset, k, trials)
    end = timer()
    return result, end-start


def choose_dataset():
    """Lets the user select a data set by keyboard input."""

    val = input('Type "r" for restaurant data set, "p" for plants data set, '
                '"b" for books data set, or anything else for business data set: ')
    if val == 'r':
        size = int(input('How many examples do you want to use? '))
        return SyntheticRestaurant(size)
    elif val == 'p':
        dataset = DataSet(attr_names='Habitat Colour TypeOfLeaf LeafWidth LeafLength Height EdibleOrPoisonous',
                          name='plants',
                          source='http://mldata.org/repository/data/viewslug/plant-classification')
    elif val == 'b':
        dataset = DataSet(attr_names='Genre MenBuyers WomenBuyers Price CriticismRate ? LikedByAudience',
                          name='books',
                          source='http://mldata.org/repository/data/viewslug/book-evaluation-complete')
    else:
        dataset = DataSet(attr_names='X1 X2 X3 X4 X5 Successful',
                          name='business',
                          source='http://mldata.org/repository/data/viewslug/successful-business')
    return choose_size(dataset)


def choose_size(dataset):
    """Asks the user to type how many examples he/she wants to work with."""

    size = int(input('How many examples do you want to use? (max = {}) '.format(len(dataset.examples))))
    shuffle(dataset.examples)
    dataset.examples = dataset.examples[0:size]
    return dataset
