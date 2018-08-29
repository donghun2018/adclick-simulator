"""

L2Lable bidding policy, interval estimation version

bids [rpc_mean] + rho [rpc_stdev]

"""

import numpy as np

from .L2LablePolicy import L2LablePolicy


class Policy_IE_L2L(L2LablePolicy):

    def __init__(self, all_attrs, possible_bids=list(range(10)), max_t=10, ts_in_n=168, rho_range=[0], randseed=12345):
        """

        :param all_attrs:
        :param possible_bids:
        :param max_t: large T
        :param ts_in_n: how many timesteps in one n
        :param rho_range: list of values, containing parameter candidates
        :param randseed:
        """

        super().__init__(all_attrs, possible_bids, max_t, ts_in_n, rho_range, randseed=randseed)

        # initialize tunable parameter for learning to learn
        self.rho = self.prng.choice(self.rho_range)

        # initialize estimates
        self.mu = []
        self.count = []
        self.sumsq = []
        for ix in range(len(self.attrs)):
            init_mu, init_stdev = self.prng.choice(self.bid_space, size=2)
            self.mu.append(init_mu)
            self.sumsq.append(init_stdev)
            self.count.append(0)

    def _update_estimates(self, attr, x):
        """
        iterative averaging of samples
        :param attr: attribute
        :param x: sample observed
        :return: None
        """
        a_ix = self.attrs.index(attr)
        mu = self.mu[a_ix]
        n = self.count[a_ix] + 1
        mu2 = 1/n * x + (n-1)/n * mu

        sumsq2 = self.sumsq[a_ix] + (x - mu) * (x - mu2)

        self.mu[a_ix] = mu2
        self.sumsq[a_ix] = sumsq2
        self.count[a_ix] = n

    def _closest_ix_to_x(self, x, vec):
        """
        returns index ix whose value vec[ix] is closest to x.

        :param x: a number
        :param vec: list of numbers
        :return: ix: vec[ix] is closest to x, in absolute value of difference
        """
        dist = [np.abs(v - x) for v in vec]
        return np.argmin(dist)

    def bid(self, attr):
        """
        finds a bid using interval estimation

        :param attr: attribute tuple. guaranteed to be found in self.attrs
        :return: a value that is found in self.bid_space
        """
        a_ix = self.attrs.index(attr)
        est_mean = self.mu[a_ix]
        est_stdev = np.sqrt(self.sumsq[a_ix] / self.count[a_ix])
        bid = est_mean + self.rho * est_stdev
        closest_bid = self.bid_space[self._closest_ix_to_x(bid, self.bid_space)]
        return closest_bid

    def learn(self, info):
        """
        learns from auctions results

        In this sample policy, it learns by keeping track of sample averages of revenue from auctions of each attribute.
        If 'revenue_per_conversion' is empty, that means I did not get any conversions in those auctions. I ignore them.

        :param info: list of results. Single result is an aggregate outcome for all auctions with a particular attr.
                     Follows the same format as output_policy_info_?????.xlsx
        :return: does not matter
        """

        super().learn(info)

        for result in info:
            if result['revenue_per_conversion'] == '':
                continue
            attr = result['attr']
            revenue_per_click = result['revenue_per_conversion'] / result['num_click']
            self._update_estimates(attr, revenue_per_click)

        return True
