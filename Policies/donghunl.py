"""
Expected revenue bidding policy

Donghun Lee 2018


Bids sample average of observed revenue

"""


import numpy as np

from .policy import Policy    # this line is needed


class Policy_donghunl(Policy):

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

        self._initialize_average_estimates()

    def _initialize_average_estimates(self):
        """
        initial average values are set up.

        I chose to initialize with a randomly chosen bid value
        :return:
        """
        self.mu = []
        self.mu_cnt = []
        for ix in range(len(self.attrs)):
            init_val = self.prng.choice(self.bid_space)
            self.mu.append(init_val)
            self.mu_cnt.append(0)

    def _update_average_estimate(self, attr, x):
        """
        iterative averaging of samples
        :param attr: attribute
        :param x: sample observed
        :return: None
        """
        a_ix = self.attrs.index(attr)
        mu = self.mu[a_ix]
        n = self.mu_cnt[a_ix] + 1
        mu2 = 1/n * x + (n-1)/n * mu
        self.mu[a_ix] = mu2
        self.mu_cnt[a_ix] = n

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
        finds a bid that is closest to the revenue sample mean of auctions with the given attr

        :param attr: attribute tuple. guaranteed to be found in self.attrs
        :return: a value that is found in self.bid_space
        """
        a_ix = self.attrs.index(attr)
        est_mean = self.mu[a_ix]
        closest_bid = self.bid_space[self._closest_ix_to_x(est_mean, self.bid_space)]
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

        for result in info:
            if result['revenue_per_conversion'] == '':
                continue
            attr = result['attr']
            revenue = result['revenue_per_conversion'] * result['num_conversion']
            self._update_average_estimate(attr, revenue)

        return True



