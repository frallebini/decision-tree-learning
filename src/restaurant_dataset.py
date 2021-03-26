"""
Main source: http://aima.cs.berkeley.edu/python/learning.html
"""


from dataset import DataSet
from decision_tree_learner import DecisionTree
from random import choice


def RestaurantDataSet(examples=None):
    """Build a DataSet of restaurant waiting examples."""
    return DataSet(name='restaurant',
                   target='Wait',
                   examples=examples,
                   attr_names='Alternate Bar Fri/Sat Hungry ' +
                              'Patrons ' +
                              'Price ' +
                              'Raining Reservation ' +
                              'Type ' +
                              'WaitEstimate ' +
                              'Wait',
                   values=[['Yes', 'No'], ['Yes', 'No'], ['Yes', 'No'], ['Yes', 'No'],
                           ['None', 'Some', 'Full'],
                           ['$', '$$', '$$$'],
                           ['Yes', 'No'], ['Yes', 'No'],
                           ['Thai', 'Italian', 'Burger', 'French'],
                           ['0-10', '10-30', '30-60', '>60'],
                           ['Yes', 'No']])


restaurant = RestaurantDataSet()


def T(attr_name, branches):
    return DecisionTree(restaurant.attr_num(attr_name), attr_name, branches)


# A decision tree for deciding whether to wait for a table at a hotel.
waiting_decision_tree = T('Patrons',
                          {'None': 'No',
                           'Some': 'Yes',
                           'Full': T('WaitEstimate',
                                     {'>60': 'No',
                                      '0-10': 'Yes',
                                      '30-60': T('Alternate',
                                                 {'No': T('Reservation',
                                                          {'Yes': 'Yes',
                                                           'No': T('Bar', {'No': 'No',
                                                                           'Yes': 'Yes'})}),
                                                  'Yes': T('Fri/Sat', {'No': 'No',
                                                                       'Yes': 'Yes'})}),
                                      '10-30': T('Hungry',
                                                 {'No': 'Yes',
                                                  'Yes': T('Alternate',
                                                           {'No': 'Yes',
                                                            'Yes': T('Raining',
                                                                     {'No': 'No',
                                                                      'Yes': 'Yes'})})})})})


def SyntheticRestaurant(n=100):
    """Generate a DataSet with n restaurant waiting examples."""

    def gen():
        example = list(map(choice, restaurant.values))
        example[restaurant.target] = waiting_decision_tree.predict(example)
        return example

    return RestaurantDataSet([gen() for _ in range(n)])
