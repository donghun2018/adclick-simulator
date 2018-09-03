"""
Bidding Policy base class

Larry Bao and Kate Wang


A policy that implements Boltzmann for the ad-click problem.
"""

import numpy as np
import operator as op
import scipy.misc as sc
from .policy import Policy    # this line is needed


class Policy_BaoWang_PS_WeGo2(Policy):

    def __init__(self, sim_param, policy_param=None, l2l_param=None):
        """
        initializes policy base class.

        :param all_attrs: list of all possible attributes
        :param possible_bids: list of all allowed bids
        :param max_t: maximum number of auction 'iter', or iteration timesteps
        :param randseed: random number seed.
        """

        super().__init__(sim_param, policy_param)
        all_attrs = sim_param['all_attrs']
        possible_bids = sim_param['possible_bids']
        max_t = sim_param['max_T']
        randseed = policy_param['randseed']

        self.bid_price = {}
        self.alpha = {} # probabilities of clicking
        self.beta = {} # probability of converting
        self.gamma = {} # revenue per conversion
        self.lamb = {} # number of auctions
        self.cumClicks = {}

        #Exogenous Information
        self.A_t = {} # number of auctions
        self.I_t = {} # number of impressions
        self.K_t = {} # number of clicks
        self.C_t = {} # number of conversions
        self.R_t = {} # revenue / conversions



        # sets of thetas
        self.theta0 = [0.4, 0.5, 0.6, 0.7, 0.8, 0.3, 0.35, 0.5]
        self.theta1 = [0.05, 0.07, 0.09, 0.1, 0.11, 0.1, 0.09, 0.09]
        self.theta2 = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.05, 0.03]
        self.theta3 = [6, 4.5, 5, 5.5, 6.5, 7, 6.5, 5]

        self.prob_win = [1/len(self.theta0)]*len(self.theta0)


        #Initialize prior and necessary arrays
        for attr in all_attrs:
            self.bid_price[attr] = self.prng.choice(self.bid_space)
            self.A_t[attr] = 0
            self.I_t[attr] = 0
            self.K_t[attr] = 0
            self.C_t[attr] = 0
            self.R_t[attr] = 0
            self.alpha[attr] = 0.3
            self.lamb[attr] = 110
            self.beta[attr] = 0.1
            self.gamma[attr] = 35
            self.cumClicks[attr] = 0

        self.attrs = all_attrs
        self.bid_space = possible_bids
        self.max_t = max_t
        self.prng = np.random.RandomState(randseed)

        #tunable parameter
        self.theta_b = 0.5
        self.w = 0.2

    def bid(self, attr):
        """
        returns a random bid, regardless of attribute

        Note how this uses internal pseudorandom number generator (self.prng) instead of np.random

        :param attr: attribute tuple. guaranteed to be found in self.attrs
        :return: a value that is found in self.bid_space
        """
        mu = np.zeros(len(self.bid_space))
        for j in range(len(mu)):
            for i in range(len(self.prob_win)):
                mu[j] = mu[j] + self.prob_win[i]*(self.gamma[attr] * self.beta[attr] - self.bid_space[j])*self.lamb[attr]*self.alpha[attr]*self.p_x(self.bid_space[j], i, attr)

        # rand = np.random.uniform(0, 1)
        rand = self.prng.uniform(0, 1)

        x = 0
        px = np.exp(np.multiply(self.theta_b, mu))
        px = np.divide(px, np.sum(px))
        px = np.cumsum(px)
        for i in range(len(mu)):
            if px[i] >= rand:
                x = i
                break

        return self.bid_space[x]


    def p_x(self, x, j, attr):
        #attr needs to be replaced with the features in attr
        y = 1/(1+np.exp(-(self.theta0[j]*x + self.theta1[j]*attr[0] + self.theta2[j]*attr[1])+self.theta3[j]))
        return y


    def p_w(self, x, j, attr):
        y = np.power(self.p_x(x, j, attr), self.I_t[attr]) * np.power(1-self.p_x(x, j, attr), self.A_t[attr] - self.I_t[attr])
        return y


    def learn(self, info):
        """
        learns from auctions results

        :param info: list of results. Single result is an aggregate outcome for all auctions with a particular attr.
                     Follows the same format as output_policy_info_?????.xlsx
        :return: does not matter
        """

        for result in info:
            attr = result['attr']
            self.A_t[attr] = result['num_auct']
            self.I_t[attr] = result['num_impression']
            self.lamb[attr] = self.lamb[attr] * self.w + self.A_t[attr] * (1 - self.w)

            # Case where we win
            if result['num_click'] > 0:
                self.K_t[attr] = result['num_click']
                self.C_t[attr] = result['num_conversion']
                self.R_t[attr] = result['revenue_per_conversion']
                self.beta[attr] = (self.beta[attr] * self.cumClicks[attr] + self.C_t[attr]) / (self.cumClicks[attr] + self.K_t[attr])
                self.cumClicks[attr] = self.cumClicks[attr] + result['num_click']
                self.alpha[attr] = self.alpha[attr] * self.w + self.K_t[attr] / self.I_t[attr] * (1 - self.w)

                for j in range(len(self.theta0)):
                    Pw = self.p_w(result['your_bid'], j, attr)
                    self.prob_win[j] = self.prob_win[j] * Pw

                self.prob_win = np.divide(self.prob_win, np.sum(self.prob_win))

                if result['num_conversion'] > 0:
                    self.gamma[attr] = self.gamma[attr] * self.w + self.R_t[attr] * (1 - self.w)

            else:
                self.I_t[attr] = 0
                for j in range(len(self.theta0)):
                    Pw = self.p_w(result['your_bid'], j, attr)
                    self.prob_win[j] = self.prob_win[j] * Pw

                self.prob_win = np.divide(self.prob_win, np.sum(self.prob_win))
        #print('lambda {} beta {} gamma {} prob_win {}'.format(self.lamb[attr], self.beta[attr], self.gamma[attr], self.prob_win))
        return True

