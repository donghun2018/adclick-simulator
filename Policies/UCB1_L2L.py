"""

L2Lable bidding policy, UCB1 algorithm based.
(no tunable parameter for UCB1)

bids [rpc_mean] + sqrt(2 ln [total_count_of_samples] / [count_of_samples_used_for_that_mean])
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                 this is the difference from RPC_bidding

Also, can choose to force initialization phase by bidding each possible action at least once before bidding like above
"""


import numpy as np

from .RPC_L2L import Policy_RPC_L2L

class Policy_UCB1_L2L(Policy_RPC_L2L):

    def __init__(self, sim_param, policy_param=None, l2l_param=None): #(self, all_attrs, possible_bids=list(range(10)), max_t=10, ts_in_n=168, rho_range=[0], randseed=12345):
        """

        """

        super().__init__(sim_param, policy_param, l2l_param)

        # initialize estimates
        self._init_rpc_mean_estimate()
        if policy_param['verbatim_UCB1'] is True:
            self.flag_init_incomplete = {a: True for a in self.attrs}
            self.verbatim_init_count = {a: [0] * len(self.bid_space) for a in self.attrs}

    def bid(self, attr):
        """
        finds a bid using UCB1 algorithm

        :param attr: attribute tuple. guaranteed to be found in self.attrs
        :return: a value that is found in self.bid_space
        """
        if self.flag_init_incomplete[attr] is True:
            prev_counts = self.verbatim_init_count[attr]
            never_tried_bid_ixs = [i for i in range(len(self.bid_space)) if prev_counts[i] == 0]
            if len(never_tried_bid_ixs) > 0:
                bid_ix = self.prng.choice(never_tried_bid_ixs)
                prev_counts[bid_ix] += 1
                if len(never_tried_bid_ixs) == 1:   # last visit in this loop
                    self.flag_init_incomplete[attr] = False

        else:
            a_ix = self.attrs.index(attr)
            est_mean = self.mu[a_ix]
            bid = est_mean + np.sqrt(2 * np.log(np.sum(self.count)) / self.count[a_ix])
            bid_ix = self._closest_ix_to_x(bid, self.bid_space)

        final_bid = self.bid_space[bid_ix]
        return final_bid

