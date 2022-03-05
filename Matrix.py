"""Matrix is a library used to take reward functions, environmental states, and actions to create reward
matrices, actions matrices, transition matrices and Q-matrices for use in dynamic programming for agent based water
models"""

import numpy as np
from collections import Counter
from itertools import product


class WaterLease:
    """Tracks the market price for leased water to be used as an opportunity cost in the planting function and
    for determining whether to plant or lease a crop"""
    def __init__(self, price):
        self.price = price

    def update_price(self, new_price):
        self.price = new_price


class Field:
    """A field object that classifies the quality of the field for use in the production function"""
    def __init__(self, quality):
        self.quality = quality


class PlantRewards:
    """A production function for the rewards a farmer receives for planting a field
    Parameters:
        -water_lease: a water lease object
        -field: a field object
        -temp: a temperature parameter"""

    def __init__(self, water_lease, field, temp):
        self.water_lease = water_lease
        self.field = field
        self.temp = temp

    def get_reward(self):
        reward = np.log((self.field.quality * self.water_lease.price)/ self.temp)
        return reward


class MarkovEnvironment:
    """An environment object that describes a markov process using a dictionary.  The dictionary keys are each possible
    environment state, the values are a second dictionary with keys that show the probability of transitioning to that
    state"""
    def __init__(self, markov_dict):
        self.markov_dict = markov_dict


class LandPortfolio:
    """Tracks the total land objects that a farmer has
    Parameters:
        land_list: a list object containing land objects
        land_actions: a vector containing a mapping of all the possible units of land that can be grown"""

    def __init__(self):
        self.land_list = []
        self.land_actions = list(range(1, len(self.land_list)+1))

    def add_field(self, field_object):
        self.land_list.append(field_object)
        self.land_actions = list(range(1, len(self.land_list)+1))


class WaterRight:
    """A water right object.  Each water right is a dict that gives you the mapping of water available based on
    markov environment"""
    def __init__(self, water_dict):
        self.water_dict = water_dict


class WaterPortfolio:
    """Maps the expected units of water available to a farmer based on the Markov environment
    Parameters:
        -water_dict: a dictionary mapping the environment to the expected units of water
        -water_actions: a vector containing a mapping of all possible water units to use"""

    def __init__(self):
        self.water_dict = {}

    def add_water_right(self, water_right):
        dict1 = Counter(self.water_dict)
        dict2 = Counter(water_right)
        self.water_dict = dict(dict1 + dict2)

    def get_water_actions(self, environment):
        """takes the natural environment and returns the number of units that could be used in that environment"""
        return list(range(1, self.water_dict[environment]+1))


class StateAction:
    """Creates a state_action vector from a water portfolio, a land portfolio and a markov process"""
    def __init__(self, water_portfolio, land_portfolio, markov):
        self.water_portfolio = water_portfolio
        self.land_portfolio = land_portfolio
        self.markov = markov
        self.action_list = []
        for key in self.markov.markov_dict:
            # get water available
            water_actions = self.water_portfolio.get_water_actions(key)
            for row in water_actions:
                water_available = list(range(1, row+1))
                temp_actions = list(product(water_available, self.land_portfolio.land_actions))
                temp_list = [key, temp_actions]
                self.action_list.append(temp_list)


if __name__ == '__main__':
    # create a land portfolio
    land = LandPortfolio()
    # create a set of fields
    for i in range(3):
        field = Field(quality=1)
        land.add_field(field) # add each new field to the field object

    # create a water portfolio
    water_portfolio = WaterPortfolio()
    for i in range(2):
        water_right = {'hot_dry': 1, 'hot_wet': 2, 'cold_dry': 1, 'cold_wet': 2}
        water_portfolio.add_water_right(water_right)

    # create the markov process
    markov_dict = {'hot_dry': {'hot_dry': 0.64, 'hot_wet': 0.16, 'cold_dry': 0.16, 'cold_wet': 0.04},
                   'hot_wet': {'hot_dry': 0.16, 'hot_wet': 0.64, 'cold_dry': 0.04, 'cold_wet': 0.16},
                   'cold_dry': {'hot_dry': 0.16, 'hot_wet': 0.04, 'cold_dry': 0.64, 'cold_wet': 0.16},
                   'cold_wet': {'hot_dry': 0.04, 'hot_wet': 0.16, 'cold_dry': 0.16, 'cold_wet': 0.64}}

    markov = MarkovEnvironment(markov_dict)
    # create the state action object
    state_vector = StateAction(water_portfolio, land, markov)
    print(state_vector.action_list)









