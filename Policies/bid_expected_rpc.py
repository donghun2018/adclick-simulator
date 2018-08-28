"""
Expected revenue-per-click bidding policy

Donghun Lee 2018


Bids closest value of sample average of observed revenue-per-click
                                                        ^^^^^^^^^^
                                                        difference from donghunl

"""


import numpy as np

from .policy import Policy    # this line is needed

from .donghunl import Policy_donghunl


class Policy_bid_expected_rpc(Policy_donghunl):

    def __init__(self, all_attrs, possible_bids=list(range(10)), max_t=10, randseed=12345):
        """
        initializes policy class.

        Note that the first line must be that super().__init__ ...

        Please use self.prng. instead of np.random. if you want to give randomness to your policy

        :param all_attrs: list of all possible attributes
        :param possible_bids: list of all allowed bids
        :param max_t: maximum number of auction 'iter', or iteration timesteps
        :param randseed: random number seed.
        """
        super().__init__(all_attrs, possible_bids, max_t, randseed=randseed)


    def learn(self, info):
        """
        learns from auctions results

        In this sample policy, it learns by keeping track of sample averages of revenue from auctions of each attribute.
        If 'revenue_per_conversion' is empty, that means I did not get any conversions in those auctions. I ignore them.

        :param info: list of results. Single result is an aggregate outcome for all auctions with a particular attr.
                     Follows the same format as output_policy_info_?????.xlsx
        :return: does not matter
        """

        for result in info:
            if result['revenue_per_conversion'] == '':
                continue
            attr = result['attr']
            revenue_per_click = result['revenue_per_conversion'] / result['num_click']
            self._update_average_estimate(attr, revenue_per_click)

        return True



