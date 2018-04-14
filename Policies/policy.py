"""
Bidding Policy base class

Donghun Lee 2018


All policy classes inherit this class.

"""

import numpy as np

class Policy():

    def __init__(self, all_attrs, possible_bids=list(range(10)), max_t=10, randseed=12345):
        """
        initializes policy base class.

        :param all_attrs: list of all possible attributes
        :param possible_bids: list of all allowed bids
        :param max_t: maximum number of auction 'iter', or iteration timesteps
        :param randseed: random number seed.
        """
        self.attrs = all_attrs
        self.bid_space = possible_bids
        self.max_t = max_t
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

