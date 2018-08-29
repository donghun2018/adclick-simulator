"""
Bidding Policy base class

Donghun Lee 2018


All policy classes inherit this class.

"""

import numpy as np

class Policy():

    def __init__(self, sim_param, policy_param=None):

        """
        initializes policy base class.

        :param all_attrs: list of all possible attributes
        :param possible_bids: list of all allowed bids
        :param max_t: maximum number of auction 'iter', or iteration timesteps
        :param randseed: random number seed.
        """
        self.attrs = sim_param['all_attrs']
        self.bid_space = sim_param['possible_bids']
        self.max_t = sim_param['max_T']
        randseed = policy_param['randseed'] if 'randseed' in policy_param else 12345
        self.prng = np.random.RandomState(randseed)

    def bid(self, attr):
        """
        returns a random bid, regardless of attribute

        Note how this uses internal pseudorandom number generator (self.prng) instead of np.random

        :param attr: attribute tuple. guaranteed to be found in self.attrs
        :return: a value that is found in self.bid_space
        """
        return self.prng.choice(self.bid_space)

    def learn(self, info):
        """
        learns from auctions results

        This policy does not learn (need not learn, because it just bids randomly)

        :param info: list of results. Single result is an aggregate outcome for all auctions with a particular attr.
                     Follows the same format as output_policy_info_?????.xlsx
        :return: does not matter
        """
        return True

