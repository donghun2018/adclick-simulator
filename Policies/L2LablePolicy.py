"""

Base class for learning-to-learn-able policy

These policies must include learning-to-learn algorithms
Default l2l algorithm is to randomly pick (no learning)

"""

from .policy import Policy


class L2LablePolicy(Policy):

    def __init__(self, all_attrs, possible_bids=list(range(10)), max_t=10, ts_in_n=168, L2L_param_range=None, randseed=12345):
        """

        :param all_attrs:
        :param possible_bids:
        :param max_t: large T
        :param ts_in_n: how many timesteps in one n
        :param L2L_param_range: list of values, containing parameter candidates
        :param randseed:
        """

        super().__init__(all_attrs, possible_bids, max_t, randseed=randseed)

        self.L2L_param_range = L2L_param_range
        self.L2L_score = [self.prng.rand() for _ in L2L_param_range]
        self.L2L_param = self.L2L_score.index(max(self.L2L_score))   # initialize L2L parameter

    def set_L2L_param_by_ix(self, rho_ix):
        """
        sets L2L parameter by its index in L2L_param_range
        :param rho_ix:
        :return:
        """
        self.L2L_param = self.L2L_param_range[rho_ix]

    def get_L2L_feedback(self):
        """
        returns g_hat := g(rho) sample for learning-to-learn algorithms
        :return:
        """

    def learnToLearn(self, rho, g_hat):
        """
        Code to learn from g(rho) =: g_hat sample.

        :param rho:
        :param g_hat:
        :return:
        """
        pass

    def update_L2L_param(self):
        """
        Updates new L2L parameter internally.
        Must be called for L2L to have effect on policy parameters
        Update policy's internal value as needed
        :return:
        """
        self.L2L_param = self.prng.choice(self.L2L_param_range)