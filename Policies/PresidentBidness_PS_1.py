import numpy as np
import copy
from scipy.stats import nbinom
from scipy.stats import norm

from .policy import Policy


class Policy_PresidentBidness_PS_1(Policy):
    def __init__(self, sim_param, policy_param=None, l2l_param=None):  # 67890):
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
        possible_bids = sim_param['possible_bids']
        self.bestEstimate = {}
        self.bid_price = {}
        self.revenue_per_click = {}
        self.mean = {}
        self.var = {}
        self.clicks = {}
        self.theta = 3
        self.profit = {}
        for attr in all_attrs:
            self.bestEstimate[attr] = self.prng.choice(self.bid_space)
            self.revenue_per_click[attr] = 0
            for bid in possible_bids:
                self.mean[(attr,bid)] = 4
                self.var[(attr,bid)] = 1
                self.clicks[(attr,bid)] = 1
                #self.theta[(attr,bid)] = 0.1
                self.profit[(attr,bid)] = []

    def bid(self, attr):
        """
        Your bidding algorithm

        Note how this uses internal pseudorandom number generator (self.prng) instead of np.random

        :param attr: attribute tuple. guaranteed to be found in self.attrs
        :return: a value that is found in self.bid_space

        k = np.size(self.bid_space)
        returns = self.bid_space - self.bestEstimate*np.ones(k)
        expval = np.zeros(k)
        for i in range(0,k):
            expval[i] = np.exp(self.theta[attr]*returns[i])
        den = np.sum(expval)
        expval = expval/den
        U = np.random.uniform(0,1)
        counter = 0
        idx = 0
        while counter < U:
            counter += expval[idx]
            idx += 1
        return self.bid_space[idx]
        """
        k = np.size(self.bid_space)
        expval = np.zeros(k)
        count = 0
        for bid in self.bid_space:
            #expval[count] = np.exp(np.exp(len(self.profit[(attr,bid)])**.4) * self.mean[(attr,bid)])
            expval[count] = np.exp(np.log(self.theta) * self.mean[(attr,bid)])
            #if count > 2:
            #    expval[count] = expval[count] *.7 + expval[count - 1] * .2 + expval[count - 2] * .1
            count += 1
        #for i in range (1,len(expval) - 1):
        #    expval[i] = expval[i]*.8 + expval[i+1]*.1 + expval[i-1]*.1
        den = np.sum(expval)
        expval = expval/den
        choice = self.prng.choice(self.bid_space,1,p=expval)
        #print(self.profit)
        if choice > 6 and len(self.profit[(attr,bid)]) < 2:
            choice = self.bid(attr)
        return choice

    def learn(self, info):
        for result in info:
            attr = result['attr']
            bid = result['your_bid']
            if result['num_click'] > 0:
                self.theta += 0.001
                rev = result['revenue_per_conversion']
                cost = result['cost_per_click']
                if isinstance(rev, str):
                    rev = 0
                pi = float(rev) - float(cost)
                self.profit[(attr,bid)].append(pi)
                #print(len(self.profit[(attr, bid)]))
                if len(self.profit[(attr,bid)])>3:
                    self.mean[(attr,bid)] = (((1 / self.var[(attr,bid)]) * (self.mean[(attr,bid)])) + ((1 / np.var(self.profit[(attr,bid)])) * pi)) / ((1. / self.var[(attr,bid)]) + 1 / np.var(self.profit[(attr,bid)]))
                    self.var[(attr,bid)] = self.var[(attr,bid)] / (1. + (self.var[(attr,bid)] / np.var(self.profit[(attr,bid)])))
                bid = round(bid+0.1,1)
                if bid < 10 and len(self.profit[(attr,bid)])>3:
                    self.mean[(attr,bid)] = (((1 / self.var[(attr,bid)]) * (self.mean[(attr,bid)])) + ((1 / np.var(self.profit[(attr,bid)])) * pi)) / ((1. / self.var[(attr,bid)]) + 1 / np.var(self.profit[(attr,bid)]))
                    self.var[(attr,bid)] = self.var[(attr,bid)] / (1. + (self.var[(attr,bid)] / np.var(self.profit[(attr,bid)])))
                    self.profit[(attr, bid)].append(pi)
                bid = round(bid + 0.1, 1)
                if bid < 10 and len(self.profit[(attr, bid)]) > 3:
                    self.mean[(attr, bid)] = (((1 / self.var[(attr, bid)]) * (self.mean[(attr, bid)])) + ( (1 / np.var(self.profit[(attr, bid)])) * pi)) / (
                                             (1. / self.var[(attr, bid)]) + 1 / np.var(self.profit[(attr, bid)]))
                    self.var[(attr, bid)] = self.var[(attr, bid)] / (1. + (self.var[(attr, bid)] / np.var(self.profit[(attr, bid)])))
                    self.profit[(attr, bid)].append(pi)
                    #print(len(self.profit[(attr, bid)]))
                bid = round(bid + 0.1, 1)
                if bid < 10 and len(self.profit[(attr, bid)]) > 3:
                    self.mean[(attr, bid)] = (((1 / self.var[(attr, bid)]) * (self.mean[(attr, bid)])) + (
                    (1 / np.var(self.profit[(attr, bid)])) * pi)) / (
                                             (1. / self.var[(attr, bid)]) + 1 / np.var(self.profit[(attr, bid)]))
                    self.var[(attr, bid)] = self.var[(attr, bid)] / (
                    1. + (self.var[(attr, bid)] / np.var(self.profit[(attr, bid)])))
                    self.profit[(attr, bid)].append(pi)
                bid = round(bid + 0.1, 1)
                if bid < 10 and len(self.profit[(attr, bid)]) > 3:
                    self.mean[(attr, bid)] = (((1 / self.var[(attr, bid)]) * (self.mean[(attr, bid)])) + (
                        (1 / np.var(self.profit[(attr, bid)])) * pi)) / (
                                                 (1. / self.var[(attr, bid)]) + 1 / np.var(self.profit[(attr, bid)]))
                    self.var[(attr, bid)] = self.var[(attr, bid)] / (
                        1. + (self.var[(attr, bid)] / np.var(self.profit[(attr, bid)])))
                bid = round(bid + 0.1, 1)
                #if bid < 10 and len(self.profit[(attr, bid)]) > 3:
                #    self.mean[(attr, bid)] = (((1 / self.var[(attr, bid)]) * (self.mean[(attr, bid)])) + (
                #        (1 / np.var(self.profit[(attr, bid)])) * pi)) / (
                #                                 (1. / self.var[(attr, bid)]) + 1 / np.var(self.profit[(attr, bid)]))
                #    self.var[(attr, bid)] = self.var[(attr, bid)] / (
                #        1. + (self.var[(attr, bid)] / np.var(self.profit[(attr, bid)])))
                bid = round(bid-0.5,1)
                if bid > 0 and len(self.profit[(attr,bid)])>3:
                    self.mean[(attr,bid)] = (((1 / self.var[(attr,bid)]) * (self.mean[(attr,bid)])) + ((1 / np.var(self.profit[(attr,bid)])) * pi)) / ((1. / self.var[(attr,bid)]) + 1 / np.var(self.profit[(attr,bid)]))
                    self.var[(attr,bid)] = self.var[(attr,bid)] / (1. + (self.var[(attr,bid)] / np.var(self.profit[(attr,bid)])))
                    self.profit[(attr, bid)].append(pi)
                bid = round(bid - 0.1, 1)
                if bid > 0 and len(self.profit[(attr, bid)]) > 3:
                    self.mean[(attr, bid)] = (((1 / self.var[(attr, bid)]) * (self.mean[(attr, bid)])) + ((1 / np.var(self.profit[(attr, bid)])) * pi)) / ((1. / self.var[(attr, bid)]) + 1 / np.var(self.profit[(attr, bid)]))
                    self.var[(attr, bid)] = self.var[(attr, bid)] / (1. + (self.var[(attr, bid)] / np.var(self.profit[(attr, bid)])))
                    self.profit[(attr, bid)].append(pi)
                bid = round(bid - 0.1, 1)
                if bid > 0 and len(self.profit[(attr, bid)]) > 3:
                    self.mean[(attr, bid)] = (((1 / self.var[(attr, bid)]) * (self.mean[(attr, bid)])) + ((1 / np.var(self.profit[(attr, bid)])) * pi)) / ((1. / self.var[(attr, bid)]) + 1 / np.var(self.profit[(attr, bid)]))
                    self.var[(attr, bid)] = self.var[(attr, bid)] / (1. + (self.var[(attr, bid)] / np.var(self.profit[(attr, bid)])))
                    self.profit[(attr, bid)].append(pi)
                bid = round(bid - 0.1, 1)
                if bid > 0 and len(self.profit[(attr, bid)]) > 3:
                    self.mean[(attr, bid)] = (((1 / self.var[(attr, bid)]) * (self.mean[(attr, bid)])) + ((1 / np.var(self.profit[(attr, bid)])) * pi)) / (
                                                 (1. / self.var[(attr, bid)]) + 1 / np.var(self.profit[(attr, bid)]))
                    self.var[(attr, bid)] = self.var[(attr, bid)] / (1. + (self.var[(attr, bid)] / np.var(self.profit[(attr, bid)])))
                #bid = round(bid - 0.1, 1)
                #if bid > 0 and len(self.profit[(attr, bid)]) > 3:
                #    self.mean[(attr, bid)] = (((1 / self.var[(attr, bid)]) * (self.mean[(attr, bid)])) + ((1 / np.var(self.profit[(attr, bid)])) * pi)) / ((1. / self.var[(attr, bid)]) + 1 / np.var(self.profit[(attr, bid)]))
                #    self.var[(attr, bid)] = self.var[(attr, bid)] / (1. + (self.var[(attr, bid)] / np.var(self.profit[(attr, bid)])))
                #self.clicksA[attr] += result['num_click']
                #self.clicksB[attr] += 1
                #lambdaSample = {}
                #PoissonVariable = {}
                #clicksAFut = {}
                #tempAttributes = list(self.attributes)
        """
        Your learning algorithm

        :param info: list of results. Single result is an aggregate outcome for all auctions with a particular attr.
                     Follows the same format as output_policy_info_?????.xlsx
        :return: does not matter
        """
        return True
