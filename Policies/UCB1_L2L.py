"""

L2Lable bidding policy, UCB1 algorithm based.
(no tunable parameter for UCB1)

bids argmax[est_profit(bid) + sqrt(2 ln [total_count_of_samples] / [count_of_samples_used_for_that_est]) ]
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                this is the difference from EPM_bidding

Also, can choose to force initialization phase by bidding each possible action at least once before bidding like above
"""
from warnings import warn

import numpy as np

from .EPM_L2L import Policy_EPM_L2L
from .L2LablePolicy import L2LablePolicy
from .RPC_L2L import Policy_RPC_L2L

class Policy_UCB1_L2L(Policy_EPM_L2L):

    def __init__(self, sim_param, policy_param=None, l2l_param=None): #(self, all_attrs, possible_bids=list(range(10)), max_t=10, ts_in_n=168, rho_range=[0], randseed=12345):
        """

        """
        self.policy_param = policy_param
        self.policy_param['verbatim_UCB1'] = True

        super().__init__(sim_param, policy_param, l2l_param)

        # initialize estimates
        self._init_counters()
        self._init_profit_estimate()
        if policy_param['verbatim_UCB1'] is True:
            self.flag_init_incomplete = {a: True for a in self.attrs}
            self.verbatim_init_count = {a: [0] * len(self.bid_space) for a in self.attrs}

    def _init_profit_estimate(self):
        self.profit = []
        for a in self.attrs:
            if self.policy_param['verbatim_UCB1'] is True:
                self.profit.append([np.inf for _ in self.bid_space])
            else:
                self.profit.append([self.prng.choice(self.bid_space) for _ in self.bid_space])

    def bid(self, attr):
        """
        finds a bid using UCB1 algorithm

        :param attr: attribute tuple. guaranteed to be found in self.attrs
        :return: a value that is found in self.bid_space
        """
        if self.policy_param['verbatim_UCB1'] is True and self.flag_init_incomplete[attr] is True:
            prev_counts = self.verbatim_init_count[attr]
            never_tried_bid_ixs = [i for i in range(len(self.bid_space)) if prev_counts[i] == 0]
            if len(never_tried_bid_ixs) > 0:
                bid_ix = self.prng.choice(never_tried_bid_ixs)
                prev_counts[bid_ix] += 1
                if len(never_tried_bid_ixs) == 1:   # last visit in this loop
                    self.flag_init_incomplete[attr] = False

        else:
            a_ix = self.attrs.index(attr)
            est_profit = self.profit[a_ix]
            counts = self.n_counter[a_ix]
            ucb1 = []
            for x_ix, x in enumerate(self.bid_space):
                if counts[x_ix] == 0:
                    ucb1.append(est_profit[x_ix])
                else:
                    ucb1.append(est_profit[x_ix] + np.sqrt(2 * np.log(np.sum(counts)) / counts[x_ix]))


            ucb1_maxval = max(ucb1)
            bid_ix = ucb1.index(ucb1_maxval)

        final_bid = self.bid_space[bid_ix]
        return final_bid

