"""

L2Lable bidding policy, interval estimation version

bids [rpc_mean] + rho [rpc_stdev]

"""

import numpy as np

from .EPM_L2L import Policy_EPM_L2L
from .RPC_L2L import Policy_RPC_L2L


class Policy_IE_L2L(Policy_EPM_L2L):

    def __init__(self, sim_param, policy_param=None, l2l_param=None): #all_attrs, possible_bids=list(range(10)), max_t=10, ts_in_n=168, rho_range=[0.01], randseed=12345):
        """

        :param all_attrs:
        :param possible_bids:
        :param max_t: large T
        :param ts_in_n: how many timesteps in one n
        :param rho_range: list of values, containing parameter candidates
        :param randseed:
        """

        super().__init__(sim_param, policy_param, l2l_param)

        # initialize estimates
        self._init_counters()
        self._init_profit_estimate_sumsq()

    def _init_profit_estimate_sumsq(self):
        self.profit_sumsq = []
        for a in self.attrs:
            self.profit_sumsq.append([0.0 for _ in self.bid_space])

    def _get_profit_est_and_stdev(self, x_ix, a_ix):
        profit = self.profit[a_ix][x_ix]
        n = self.n_counter[a_ix][x_ix]
        if n < 2:
            profit_stdev = 0.0
        else:
            profit_stdev = np.sqrt(self.profit_sumsq[a_ix][x_ix] / n)
        return profit, profit_stdev

    def bid(self, attr):
        """
        finds a bid using UCB1 algorithm

        :param attr: attribute tuple. guaranteed to be found in self.attrs
        :return: a value that is found in self.bid_space
        """

        a_ix = self.attrs.index(attr)

        score = []
        for x_ix, x in enumerate(self.bid_space):
            est_profit, profit_stdev = self._get_profit_est_and_stdev(x_ix, a_ix)
            score.append(est_profit + self.rho * profit_stdev)

        if self.policy_param['random_tiebreak'] is True:
            bid_ix, ties = Policy_EPM_L2L._argmaxr(score, self.prng)
        else:
            bid_ix = score.index(max(score))
        return bid_ix

    def _update_profit_estimate(self, bid, attr, profit):
        """
        iterative averaging of samples
        :param attr: attribute
        :param y_hat: sample observed
        :return: None
        """
        a_ix = self.attrs.index(attr)
        x_ix = self.bid_space.index(bid)
        n = self.n_counter[a_ix][x_ix]
        mu = self.profit[a_ix][x_ix]

        if n == 0:
            mu2 = profit
            sumsq2 = 0.0
        else:
            mu2 = n / (n+1) * mu + 1 / (n+1) * profit
            sumsq2 = self.profit_sumsq[a_ix][x_ix] + (profit - mu) * (profit - mu2)

        self.profit[a_ix][x_ix] = mu2
        self.profit_sumsq[a_ix][x_ix] = sumsq2
        self.n_counter[a_ix][x_ix] = n + 1
