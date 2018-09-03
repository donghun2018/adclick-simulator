import numpy as np
import copy
from scipy.stats import nbinom
from scipy.stats import norm

from .policy import Policy


class Policy_PresidentBidness_LA_1(Policy):
    def __init__(self, sim_param, policy_param=None, l2l_param=None): # 67890):
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
        self.bid_price = {}
        self.revenue_per_click = {}
        self.clicks = {}
        self.clicksA = {}
        self.clicksB = {}
        self.numConvA = {}
        self.numConvB = {}
        self.RevMean = {}
        self.RevVar = {}
        self.Rev = {}
        self.attributes = all_attrs
        self.KGRevenue = {}
        self.KGClick = {}
        self.KGConv = {}
        self.bestEstimate = {}
        for attr in all_attrs:
            self.bestEstimate[attr] = self.prng.choice(self.bid_space)
            self.revenue_per_click[attr] = 0
            self.clicks[attr] = 1
            self.clicksA[attr] = 30
            self.clicksB[attr] = 2
            self.numConvA[attr] = 3
            self.numConvB[attr] = 3
            self.RevMean[attr] = 4
            self.RevVar[attr] = .2
            self.Rev[attr] = []
            self.KGRevenue[attr] = 1
            self.KGClick[attr] = 0.5
            self.KGConv[attr] = 1

    def bid(self, attr):
        """
        Your bidding algorithm

        Note how this uses internal pseudorandom number generator (self.prng) instead of np.random

        :param attr: attribute tuple. guaranteed to be found in self.attrs
        :return: a value that is found in self.bid_space
        """

        idx = (np.abs(self.bid_space - self.bestEstimate[attr])).argmin()
        if self.bid_space[idx] > 7.5:
            bidReturn = 7.5
        else:
            bidReturn = self.bid_space[idx]
        return bidReturn

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
                self.clicksA[attr] += result['num_click']
                self.clicksB[attr] += 1
                lambdaSample = {}
                PoissonVariable = {}
                clicksAFut = {}
                tempAttributes = list(self.attributes)

                for attrClick in tempAttributes:
                    lambdaSample[attrClick] = self.prng.gamma(self.clicksA[attrClick], 1 / self.clicksB[attrClick], 100)
                    PoissonVariable[attrClick] = self.prng.poisson(lambdaSample[attrClick])
                    clicksAFut[attrClick] = self.clicksA[attrClick] + PoissonVariable[attrClick]
                    clicksAFut[attrClick] = clicksAFut[attrClick] / (self.clicksB[attrClick] + 1)
                    self.KGClick[attrClick] = clicksAFut[attrClick] - (
                    self.clicksA[attrClick] / self.clicksB[attrClick])
                    self.KGClick[attrClick] = np.mean(self.KGClick[attrClick])
                # print(self.KGClick)
                # print({k: ((float(self.clicksA[k])/self.clicksB[k])) for k in self.clicksA})
                self.numConvA[attr] += result['num_conversion']
                self.numConvB[attr] += 1
                lambdaSample = {}
                PoissonVariable = {}
                numConvAFut = {}
                tempAttributes = list(self.attributes)
                for attrConv in tempAttributes:
                    lambdaSample[attrConv] = self.prng.gamma(self.numConvA[attrConv], 1 / self.numConvB[attrConv], 100)
                    PoissonVariable[attrConv] = self.prng.poisson(lambdaSample[attrConv])
                    numConvAFut[attrConv] = self.numConvA[attrConv] + PoissonVariable[attrConv]
                    numConvAFut[attrConv] = numConvAFut[attrConv] / (self.numConvB[attrConv] + 1)
                    self.KGConv[attrConv] = numConvAFut[attrConv] - (self.numConvA[attrConv] / self.numConvB[attrConv])
                    self.KGConv[attrConv] = max(-0.00001, np.mean(self.KGConv[attrConv]))
                # print(PoissonVariable)
                # print({k: ((float(self.numConvA[k])/self.numConvB[k])) for k in self.numConvA})
                # print(self.KGConv)
                # print(numConvAFut)
                # print(self.KGConv)
                if result['num_conversion'] > 0:
                    self.Rev[attr].append(np.log(result['revenue_per_conversion']))
                    for attrRev in tempAttributes:
                        if len(self.Rev[attrRev]) > 2:
                            # print(self.Rev[attr])
                            # print(self.RevVar[attr])
                            self.RevMean[attrRev] = (((1 / self.RevVar[attrRev]) * (self.RevMean[attrRev])) + (
                            (1 / np.var(self.Rev[attrRev])) * (np.log(result['revenue_per_conversion'])))) / (
                                                    (1. / self.RevVar[attrRev]) + 1 / np.var(self.Rev[attrRev]))
                            # print(self.RevMean[attr])
                            self.RevVar[attrRev] = self.RevVar[attrRev] / (
                            1. + (self.RevVar[attrRev] / np.var(self.Rev[attrRev])))
                            varKG = self.RevVar[attrRev] / (1 + np.var(self.Rev[attrRev]) / self.RevVar[attrRev])
                            # print(varKG)
                            otherMaxAtt = dict(self.RevMean)
                            del otherMaxAtt[attrRev]
                            maxKey = max(otherMaxAtt.keys(), key=(lambda k: otherMaxAtt[k]))
                            maxValue = otherMaxAtt[maxKey]
                            zeta = -np.abs((self.RevMean[attrRev] - maxValue) / np.power(varKG, 0.5))
                            self.KGRevenue[attrRev] = (
                            zeta * norm.cdf(zeta) + norm.pdf(zeta))  # + 1 / len(self.Rev[attrRev])
                            # print(self.KGRevenue)
                            # print(KGClick)
                            # print(KGConv)
                            # print(self.Rev)
                            # print(np.var(self.Rev))
        # print({k: float(self.numConvA[k])/self.numConvB[k] + self.KGConv[k] for k in self.clicksA})
        # print(self.numConvB)
        self.bestEstimate = {k: ((float(self.numConvA[k]) / self.numConvB[k] + self.KGConv[k]) / (
        float(self.clicksA[k]) / self.clicksB[k] + self.KGClick[k])) * np.exp(self.RevMean[k] + self.KGRevenue[k]) for k
                             in self.clicksA}
        # print(self.bestEstimate)
        return True

