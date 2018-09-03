"""

L2Lable bidding policy, estimated-profit-maximizing bid

bids argmax[est_profit(bid)]


"""


import numpy as np

from .L2LablePolicy import L2LablePolicy


class Policy_EPM_L2L(L2LablePolicy):

    def __init__(self, sim_param, policy_param=None, l2l_param=None):
        """

        """

        super().__init__(sim_param, policy_param, l2l_param)
        self.policy_param = policy_param
        self.policy_param['force_initial_exploration'] = False
        self.policy_param['random_tiebreak'] = True

        # initialize estimates
        self._init_counters()
        self._init_profit_estimate()

    def _init_counters(self):
        self.n_counter = []
        for a in self.attrs:
            self.n_counter.append([0 for _ in self.bid_space])

    def _init_profit_estimate(self):
        self.profit = []
        for a in self.attrs:
            if self.policy_param['force_initial_exploration'] is True:
                self.profit.append([np.inf for _ in self.bid_space])
            else:
                self.profit.append([self.prng.choice(self.bid_space) for _ in self.bid_space])

    def _update_profit_estimate(self, bid, attr, profit):
        bid_ix = self.bid_space.index(bid)
        attr_ix = self.attrs.index(attr)
        n = self.n_counter[attr_ix][bid_ix]
        mu = self.profit[attr_ix][bid_ix]

        if n == 0:
            mu2 = profit
        else:
            mu2 = n / (n+1) * mu + 1 / (n+1) * profit

        self.profit[attr_ix][bid_ix] = mu2
        self.n_counter[attr_ix][bid_ix] = n+1

    @staticmethod
    def _argmaxr(l, prng=None):
        """
        returns index of max element in l, with random tiebreaking
        :param l: list
        :param prng: pseudorandom number generator from numpy (np.random.RandomState)
        :return:
        """
        max_val = max(l)
        max_ix = [i for i in range(len(l)) if l[i] == max_val]
        if prng:
            argmaxr = prng.choice(max_ix)
        else:
            argmaxr = np.random.choice(max_ix)
        return np.asscalar(argmaxr), max_ix

    def bid(self, attr):
        """

        :param attr: attribute tuple. guaranteed to be found in self.attrs
        :return: a value that is found in self.bid_space
        """
        a_ix = self.attrs.index(attr)
        est_profit = self.profit[a_ix]
        if self.policy_param['random_tiebreak'] is True:
            bid_ix, ties = Policy_EPM_L2L._argmaxr(est_profit, self.prng)
        else:
            bid_ix = est_profit.index(max(est_profit))
        return bid_ix

    def learn(self, info):

        super().learn(info)

        for result in info:
            attr = result['attr']
            if result['revenue_per_conversion'] == '':
                profit = 0.0
            else:
                revenue = result['revenue_per_conversion'] * result['num_conversion']
                cost = result['cost_per_click'] * result['num_click']
                profit = revenue - cost
            my_bid = result['your_bid']
            self._update_profit_estimate(my_bid, attr, profit)

