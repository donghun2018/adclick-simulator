"""

Base class for learning-to-learn-able policy

These policies must include learning-to-learn algorithms
Default l2l algorithm is to randomly pick (no learning)

"""

import numpy as np

from .policy import Policy


class L2LablePolicy(Policy):

    def __init__(self, all_attrs, possible_bids=list(range(10)), max_t=10, ts_in_n=168, rho_range=[0.5, 1., 1.5, 2.0], randseed=12345):
        """

        :param all_attrs:
        :param possible_bids:
        :param max_t: large T
        :param ts_in_n: how many timesteps in one n
        :param rho_range: list of values, containing parameter candidates
        :param randseed:
        """

        super().__init__(all_attrs, possible_bids, max_t, randseed=randseed)

        self.rho_range = rho_range
        self.rho = self.prng.choice(rho_range)   # initialize L2L parameter

        self.sum_C = 0
        self.sum_C_count = 0

    def set_rho_by_ix(self, rho_ix):
        """
        sets L2L parameter by its index in rho_range
        :param rho_ix:
        :return:
        """
        self.rho = self.rho_range[rho_ix]

    def _record_g_hat(self, C_hat):
        self.sum_C += C_hat
        self.sum_C_count += 1

    def get_L2L_feedback(self):
        """
        returns g_hat := g(rho) sample for learning-to-learn algorithms
        :return:
        """
        g_hat = self.sum_C
        self.sum_C = self.sum_C_count = 0 # reset counter
        return g_hat, self.rho

    def learn(self, info):
        """
        learns from auctions results

        This includes learning-to-learn prep work.

        :param info: list of results. Single result is an aggregate outcome for all auctions with a particular attr.
                     Follows the same format as output_policy_info_?????.xlsx
        :return: does not matter
        """

        super().learn(info)

        profit = 0
        for result in info:
            cost = 0
            if result['num_click'] != 0:
                cost = result['num_click'] * result['cost_per_click']
            revenue = 0
            if result['num_conversion'] != 0:
                revenue = result['num_conversion'] * result['revenue_per_conversion']
            profit += revenue - cost
        self._record_g_hat(profit)

        return True
