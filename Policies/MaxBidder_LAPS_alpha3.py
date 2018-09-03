import numpy as np
from scipy.special import comb
from scipy.stats import norm

from .policy import Policy

class Policy_MaxBidder_LAPS_alpha3(Policy):

    def __init__(self, sim_param, policy_param=None, l2l_param=None):
        """
        Your bidding policy class initializer

        Note that the first line must be that super().__init__ ...

        Please use self.prng. instead of np.random. if you want to give randomness to your policy

        :param all_attrs: list of all possible attributes
        :param possible_bids: list of all allowed bids
        :param max_t: maximum number of auction 'iter', or iteration timesteps
        :param randseed: random number seed.
        """
        super().__init__(sim_param, policy_param)
        all_attrs = sim_param['all_attrs']
        num_attrs = len(all_attrs)
        num_bids = len(self.bid_space)
        self.iter = 1
        self._init_vars(num_attrs, num_bids)

    def _init_vars(self, num_attrs, num_bids):
        self.num_attrs = num_attrs
        self.num_bids = num_bids
        num_thetas = 11
        self.thetas = np.array([np.linspace(0.5, 1.5, num=num_thetas) for i in range(num_attrs)])
        self.p = np.array([[1.0/num_thetas for i in range(num_thetas)] for j in range(num_attrs)])
        self.priors = [{'num_auct': 100,
                       'prob_click': 0.5,
                       'prob_conversion': 0.15,
                       'revenue_per_conversion': 50} for i in range(self.num_attrs)]
        bias = np.repeat(0.1, self.num_bids)
        self.cost_bias = np.array([bias.copy() for j in range(self.num_attrs)])
        self.boost = 0.2
        self.learn_steps = 20
        self.bid_idx = {x:i for i, x in enumerate(self.bid_space)}

    def _update(self, W):
        attr_idx = self.attrs.index(W['attr'])
        bid = W['your_bid']
        bid_idx = self.bid_idx[bid]
        win = int(bid == W['winning_bid'])
        theta_p = list(zip(self.thetas[attr_idx], self.p[attr_idx]))
        c_w = np.sum([p * self._f_w(self.priors[attr_idx]['num_auct'], self._logistic(bid, theta) * self.priors[attr_idx]['num_auct'], theta, bid) for theta, p in theta_p])
        self.p[attr_idx] = np.array([p * self._f_w(self.priors[attr_idx]['num_auct'], self._logistic(bid, theta) * self.priors[attr_idx]['num_auct'], theta, bid) for theta, p in theta_p]) / c_w
        self._point_est_update('num_auct', attr_idx, W['num_auct'])
        if win:
            if W['num_click'] > 0:
                self._point_est_update('prob_click', attr_idx, W['num_click'] / W['num_impression'])
                self._point_est_update('prob_conversion', attr_idx, W['num_conversion'] / W['num_click'])
                self._point_est_update('revenue_per_conversion', attr_idx, W['revenue_per_conversion'])
                self._bias_update(attr_idx, bid_idx, bid - W['cost_per_click'])
        else:
            if self.iter < self.learn_steps:
                self.cost_bias[attr_idx][(np.array(self.bid_space) > bid + 0.5) & (np.array(self.bid_space) <= bid + 1.5)] += self.boost / self.iter

    def _point_est_update(self, prior, attr_idx, W):
        W = W if W else 0
        mean = self.priors[attr_idx][prior]
        self.priors[attr_idx][prior] = (self.iter * mean + W) / (self.iter + 1)

    def _bias_update(self, attr_idx, bid_idx, W):
        mean = self.cost_bias[attr_idx][bid_idx]
        self.cost_bias[attr_idx][bid_idx] = (self.iter * mean + W) / (self.iter + 1)

    def _logistic(self, x, theta):
        x -= np.mean(self.bid_space)
        return 1 / (1 + np.exp(-theta * x))

    def _expected_profit(self, attr_idx, bid_space):
        mu = np.zeros(len(bid_space))
        revenue = self.priors[attr_idx]['revenue_per_conversion'] * self.priors[attr_idx]['prob_conversion']
        for idx, bid in enumerate(bid_space):
            bid_idx = self.bid_idx[bid]
            mu[idx] = np.sum([p * (revenue - (bid - self.cost_bias[attr_idx][bid_idx])) * self.priors[attr_idx]['prob_click'] * self.priors[attr_idx]['num_auct'] * self._logistic(bid, theta) for theta, p in zip(self.thetas[attr_idx], self.p[attr_idx])])
        return mu

    def _expected_profit_single(self, attr_idx, bid_idx, num_auct, p, theta):
        bid = self.bid_space[bid_idx]
        revenue = self.priors[attr_idx]['revenue_per_conversion'] * self.priors[attr_idx]['prob_conversion']
        return p * (revenue - (bid - self.cost_bias[attr_idx][bid_idx])) * self.priors[attr_idx]['prob_click'] * num_auct * self._logistic(bid, theta)

    def _f_w(self, num_auct, num_impression, theta, bid):
        p = self._logistic(bid, theta)
        return comb(num_auct, num_impression) * p**num_impression * (1 - p)**(num_auct - num_impression)

    def _kg(self, thetas, ps, attr_idx):
        bid_space = np.around(np.arange(4.0, 6.0, step=0.2), 1)
        bid_idx = [self.bid_idx[x] for x in bid_space]
        v_kg = np.zeros(len(bid_space))
        mean_num_auct = self.priors[attr_idx]['num_auct']
        mu = self._expected_profit(attr_idx, bid_space)
        X = 15
        L = 30
        for idx, bid in enumerate(bid_space):
            v_kg_x = 0
            prob_win = np.sum([p * self._logistic(bid, theta) for theta, p in zip(thetas, ps)])
            gen_auct = [self.prng.poisson(mean_num_auct) for l in range(L)]
            gen_imp = [self.prng.binomial(num_auct, prob_win) for num_auct in gen_auct]
            c_w = [np.sum([p * self._f_w(a, i, theta, bid) for theta, p in zip(thetas, ps)]) for a, i in zip(gen_auct, gen_imp)]
            for l in range(L):
                num_auct = gen_auct[l]
                num_impression = gen_imp[l]
                v_kg_x_max = []
                for i in range(X):
                    x_idx = self.prng.choice(bid_idx)
                    v_kg_x_max.append(1.0 / c_w[l] * np.sum([p * self._f_w(num_auct, num_impression, theta, bid) * self._expected_profit_single(attr_idx, x_idx, num_auct, p, theta) for theta, p in zip(thetas, ps)]))
                v_kg_x += np.amax(v_kg_x_max)
            v_kg[idx] = v_kg_x / L
        v_kg = v_kg - mu
        x = np.argmax(mu + (self.max_t - self.iter) * v_kg)
        return bid_idx[x]

    def bid(self, attr):
        """
        Your bidding algorithm

        Note how this uses internal pseudorandom number generator (self.prng) instead of np.random

        :param attr: attribute tuple. guaranteed to be found in self.attrs
        :return: a value that is found in self.bid_space
        """
        attr_idx = self.attrs.index(attr)
        thetas = self.thetas[attr_idx]
        ps = self.p[attr_idx]
        x = self._kg(thetas, ps, attr_idx)
        return self.bid_space[x]

    def learn(self, info):
        """
        Your learning algorithm

        :param info: list of results. Single result is an aggregate outcome for all auctions with a particular attr.
                     Follows the same format as output_policy_info_?????.xlsx
        :return: does not matter
        """
        for result in info:
            self._update(result)
        self.iter += 1
        return True
