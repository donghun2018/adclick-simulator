"""

L2Lable bidding policy, UCB1 algorithm based.
(no tunable parameter for UCB1)

bids [rpc_mean] + sqrt(2 ln [total_count_of_samples] / [count_of_samples_used_for_that_mean])
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                 this is the difference from RPC_bidding

"""


import numpy as np

from .RPC_L2L import Policy_RPC_L2L

class Policy_UCB1_L2L(Policy_RPC_L2L):

    def __init__(self, sim_param, policy_param=None, l2l_param=None): #(self, all_attrs, possible_bids=list(range(10)), max_t=10, ts_in_n=168, rho_range=[0], randseed=12345):
        """

        :param all_attrs:
        :param possible_bids:
        :param max_t: large T
        :param ts_in_n: how many timesteps in one n
        :param rho_range: list of values, containing parameter candidates
        :param randseed:
        """

        super().__init__(sim_param, policy_param, l2l_param)
        # all_attrs, possible_bids, max_t, ts_in_n, rho_range, randseed=randseed)

        # initialize estimates
        self._init_rpc_mean_estimate()

    def bid(self, attr):
        """
        finds a bid using interval estimation

        :param attr: attribute tuple. guaranteed to be found in self.attrs
        :return: a value that is found in self.bid_space
        """
        a_ix = self.attrs.index(attr)
        est_mean = self.mu[a_ix]
        bid = est_mean + np.sqrt(2 * np.log(np.sum(self.count)) / self.count[a_ix])
        closest_bid = self.bid_space[self._closest_ix_to_x(bid, self.bid_space)]
        return closest_bid

