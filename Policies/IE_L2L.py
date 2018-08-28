"""

L2Lable bidding policy, interval estimation version

bids [rpc_mean] + rho [rpc_stdev]

"""



from .L2LablePolicy import L2LablePolicy

class Policy_IE_L2L(L2LablePolicy):

    def __init__(self, all_attrs, possible_bids=list(range(10)), max_t=10, ts_in_n=168, L2L_param_range=None, randseed=12345):
        """

        :param all_attrs:
        :param possible_bids:
        :param max_t: large T
        :param ts_in_n: how many timesteps in one n
        :param L2L_param_range: list of values, containing parameter candidates
        :param randseed:
        """

        super().__init__(all_attrs, possible_bids, max_t, ts_in_n, L2L_param_range, randseed=randseed)

    def learnToLearn(self, rho, g_hat):
        """
        learn from g(rho) =: g_hat sample.

        :param rho:
        :param g_hat:
        :return:
        """
        rho_ix = self.L2L_param_range.index(rho)
        self.L2L_score[rho_ix] = g_hat

    def update_L2L_param(self):
        """
        Updates new L2L parameter internally.
        Must be called for L2L to have effect on policy parameters
        :return:
        """
        self.L2L_param = self.L2L_score.index(max(self.L2L_score))