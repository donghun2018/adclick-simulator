import numpy as np

from .policy import Policy


class Policy_whan(Policy):

    def __init__(self, all_attrs, possible_bids=range(10), max_t=10, randseed=67890):
        """
        Your bidding policy class initializer

        Note that the first line must be that super().__init__ ...

        Please use self.prng. instead of np.random. if you want to give randomness to your policy

        :param all_attrs: list of all possible attributes
        :param possible_bids: list of all allowed bids
        :param max_t: maximum number of auction 'iter', or iteration timesteps
        :param randseed: random number seed.
        """
        super().__init__(all_attrs, possible_bids, max_t, randseed=randseed)
        self.bid_price = {}
        self.revenue_per_click = {}
        self.clicks = {}
        for attr in all_attrs:
            self.bid_price[attr] = self.prng.choice(self.bid_space)
            self.revenue_per_click[attr] = 0
            self.clicks[attr] = 0

    def bid(self, attr):
        """
        Your bidding algorithm

        Note how this uses internal pseudorandom number generator (self.prng) instead of np.random

        :param attr: attribute tuple. guaranteed to be found in self.attrs
        :return: a value that is found in self.bid_space
        """
        return self.bid_price[attr]

    def learn(self, info):
        """
        Your learning algorithm

        :param info: list of results. Single result is an aggregate outcome for all auctions with a particular attr.
                     Follows the same format as output_policy_info_?????.xlsx
        :return: does not matter
        """
        for result in info:
            attr = result['attr']
            if result['num_click'] > 0:
                old_revenue = self.revenue_per_click[attr]*self.clicks[attr]
                if result['num_conversion'] > 0:
                    add_revenue = result['num_conversion']*result['revenue_per_conversion']
                else:
                    add_revenue = 0
                self.clicks[attr] += result['num_click']
                self.revenue_per_click[attr] = (old_revenue+add_revenue)/self.clicks[attr]
            if result['winning_bid'] > self.revenue_per_click[attr]:
                higher_than_revenue_bids = [p for p in self.bid_space if p > self.revenue_per_click[attr]]
                # choose the smallest bid from the bids that are higher than the revenue
                if len(higher_than_revenue_bids) > 0:
                    self.bid_price[attr] = np.min(higher_than_revenue_bids)
                else:
                    self.bid_price[attr] = np.max(self.bid_space)
            else:
                higher_than_winning_bids = [p for p in self.bid_space if p > result['winning_bid']]
                # choose the smallest bid from the bids that are higher than the winning bid
                if len(higher_than_winning_bids) > 0:
                    self.bid_price[attr] = np.min(higher_than_winning_bids)
                else:
                    self.bid_price[attr] = np.max(self.bid_space)
        return True

