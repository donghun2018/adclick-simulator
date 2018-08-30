"""

Base class for learning-to-learn-able policy

These policies must include learning-to-learn algorithms
Default l2l algorithm is to randomly pick (no learning)

"""

import numpy as np

from .policy import Policy


class L2LablePolicy(Policy):

    def __init__(self, sim_param, policy_param=None, l2l_param=None):#  ts_in_n=168, rho_range=[0.5, 1., 1.5, 2.0], randseed=12345):
        """

        :param all_attrs:
        :param possible_bids:
        :param max_t: large T
        :param ts_in_n: how many timesteps in one n
        :param rho_range: list of values, containing parameter candidates
        :param randseed:
        """

        super().__init__(sim_param, policy_param)

        self.rho_range = l2l_param['rho_range'] if l2l_param else [0]
        self.rho = self.prng.choice(self.rho_range)   # initialize L2L parameter

        self.sum_C = 0
        self.sum_C_count = 0

    def set_rho_by_ix(self, rho_ix):
        """
        sets L2L parameter by its index in rho_range
        :param rho_ix:
        :return:
        """
        self.rho = self.rho_range[rho_ix]

    def _record_g_hat(self, info):
        """
        records C_hat values (profit) to generate g_hat
        :param info:
        :return:
        """
        profit = 0
        for result in info:
            cost = 0
            if result['num_click'] != 0:
                cost = result['num_click'] * result['cost_per_click']
            revenue = 0
            if result['num_conversion'] != 0:
                revenue = result['num_conversion'] * result['revenue_per_conversion']
            profit += revenue - cost
        self.sum_C += profit
        self.sum_C_count += 1

    def get_L2L_feedback(self):
        """
        returns g_hat := g(rho) sample for learning-to-learn algorithms
        :return:
        """
        g_hat = self.sum_C
        t = self.sum_C_count
        self.sum_C = self.sum_C_count = 0 # reset counter
        return g_hat, self.rho, t

    def learn(self, info):
        """
        learns from auctions results

        This includes learning-to-learn prep work.

        :param info: list of results. Single result is an aggregate outcome for all auctions with a particular attr.
                     Follows the same format as output_policy_info_?????.xlsx
        :return: does not matter
        """

        super().learn(info)
        self._record_g_hat(info)

        return True
