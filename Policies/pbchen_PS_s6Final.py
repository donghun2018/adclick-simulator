"""
Patrick Chen
May 14 2018 PS Final
"""


import numpy as np
import math as math

from .policy import Policy    # this line is needed


class Policy_pbchen_PS_s6Final(Policy):

    def __init__(self, sim_param, policy_param=None, l2l_param=None):
        """
        initializes policy class.

        Note that the first line must be that super().__init__ ...

        Please use self.prng. instead of np.random. if you want to give randomness to your policy

        :param all_attrs: list of all possible attributes
        :param possible_bids: list of all allowed bids
        :param max_t: maximum number of auction 'iter', or iteration timesteps
        :param randseed: random number seed.
        """
        super().__init__(sim_param, policy_param)

        self._initialize_state_var()

    def _initialize_state_var(self):
        """
        initial parameters are set up
        """
        self.max_rounds = 168
        stable = 5
        self.n = 1
        self.num_auction = []
        self.rev = []

        self.num_click = np.ones(len(self.attrs))*stable
        self.num_conv = np.ones(len(self.attrs))*stable*.1 # p convert = .1
        self.num_impr = np.ones(len(self.attrs))*stable  * 3 # pclick = .33

        x = 2 # round(len(self.bid_space)/10)
        no_low = np.ones((len(self.attrs),x))* self.max_rounds

        self.n_used = np.concatenate((no_low, 
            np.ones((len(self.attrs),len(self.bid_space) - x))), axis = 1)

        self.n_theta = 100
        self.theta = [] 
        self.theta_belief = []

        self.ucb_param = 5 #5 Best out of 2-6
        self.explore_counter = 1
        self.explore_profit = 0

        for i in range(len(self.attrs)):
            self.num_auction.append(120)
            self.rev.append(50)

            # Generate thetas
            const = self.prng.normal(5, 2, size = self.n_theta)
            price_sensitivity = self.prng.normal(1, .2, size = self.n_theta)
            gen_thetas = np.column_stack((const, price_sensitivity))
            self.theta.append(gen_thetas)

            # Uniform initial beliefs about thetas
            initial_theta_belief =  np.ones(self.n_theta)/self.n_theta
            self.theta_belief.append(initial_theta_belief)

    def _calc_presult(self, attr, bid, result):
        a_ix = self.attrs.index(attr)
        params = self.theta[a_ix]
        presult = []

        for theta in params:
            p_win = 1/(1+math.exp(np.sum(theta * np.array([1, -bid]))))
            if result == 0:
                p_iresult = 1 - p_win
            else:
                p_iresult = p_win
            presult.append(p_iresult)
        return presult

    def get_prior(self):
        values = []

        values.append(self.num_auction)
        values.append(self.num_click/self.num_impr)
        values.append(self.num_conv/self.num_click)
        values.append(self.rev)

        best_theta0 = []
        best_thetax = []
        for i in range(len(self.attrs)):
            a_ix = np.argmax(self.theta_belief[i])
            best_theta = self.theta[i][a_ix]
            best_theta0.append(best_theta[0])
            best_thetax.append(best_theta[1])

        best_theta = []
        best_theta.append(np.mean(best_theta0))
        best_theta.append(np.std(best_theta0)) 
        best_theta.append(np.mean(best_thetax))
        best_theta.append(np.std(best_thetax))
        values.append(best_theta) # mean theta0, std theta0, mean thetax, std thetax
        return(values)

    def bid(self, attr):
        """
        finds a bid that is closest to the revenue sample mean of auctions with the given attr

        :param attr: attribute tuple. guaranteed to be found in self.attrs
        :return: a value that is found in self.bid_space
        """
        a_ix = self.attrs.index(attr)
        exp_profit = []

        for i in range(len(self.bid_space)):
            bid = self.bid_space[i]

            p_win_theta = self._calc_presult(attr, bid, 1) # each theta belief about winning


            if bid <= 3:
                cost = bid
            elif bid <= 5.5:
                cost = (bid-3)*.2+3
            else:
                cost = bid - 2


            pred_pwin = p_win_theta * self.theta_belief[a_ix] # weight by belief about theta

            profit_from_bid = (self.num_auction[a_ix] * (self.num_click[a_ix]/self.num_impr[a_ix]) * 
                (self.rev[a_ix]*(self.num_conv[a_ix]/self.num_click[a_ix]) - cost) * sum(pred_pwin))
            exp_profit.append(profit_from_bid)

        explore = (self.ucb_param*np.power((2*math.log(self.n)/self.n_used[a_ix]),0.5))

        # a = np.diff(exp_profit)
        # print("explore", np.max(explore), np.min(explore), np.mean(explore), "/ prof: ",
        #     np.max(a), np.min(a), np.mean(a))

        bid = np.argmax(exp_profit+explore) # Find best bid
        self.n_used[a_ix][bid] = self.n_used[a_ix][bid] +1 # Update used counter

        if np.argmax(exp_profit) != bid:
            self.explore_counter = self.explore_counter+1

        return(self.bid_space[bid])

    def learn(self, info):
        """
        learns from auctions results

        In this sample policy, it learns by keeping track of sample averages of revenue from auctions of each attribute.
        If 'revenue_per_conversion' is empty, that means I did not get any conversions in those auctions. I ignore them.

        :param info: list of results. Single result is an aggregate outcome for all auctions with a particular attr.
                     Follows the same format as output_policy_info_?????.xlsx
        :return: does not matter
        """
        self.n = self.n+1

        for result in info:
            # Get attribute index
            a_ix = self.attrs.index(result['attr'])

            # Update number of expected auctions
            self.num_auction[a_ix] = ((1-1/self.n)*self.num_auction[a_ix] 
                + 1/self.n*result['num_auct'])

            # Calculate probability of auction result
            p_auc_result = self._calc_presult(result['attr'],
                result['your_bid'], int(result['your_bid'] == result['winning_bid']))
            # Update p(True theta)
            self.theta_belief[a_ix] = p_auc_result * self.theta_belief[a_ix]  
            self.theta_belief[a_ix] = self.theta_belief[a_ix]/sum(self.theta_belief[a_ix])

            # update number of clicks
            self.num_click[a_ix] = self.num_click[a_ix] + result['num_click']
            # update number of impressions
            self.num_impr[a_ix] = self.num_impr[a_ix] + result['num_impression']

            # update number of conversions
            old_num_conv = self.num_conv[a_ix]
            self.num_conv[a_ix] = self.num_conv[a_ix]  + result['num_conversion']

            # Update expected revenue
            if result['revenue_per_conversion'] != '':
                self.rev[a_ix] = ((self.rev[a_ix]*old_num_conv + 
                    result['revenue_per_conversion'] * result['num_conversion'])/
                    self.num_conv[a_ix])

                self.explore_profit = self.explore_profit + (
                    result['revenue_per_conversion'] * result['num_conversion'] - 
                    result['num_click'] * result['cost_per_click'])

            if self.n % 10 == 0:
                if self.explore_counter > 50:
                    if self.explore_profit > 0:
                        sign = 1
                    else:
                        sign = -1
                else:
                    sign = 0

                delta = 2*10/168*sign
                self.ucb_param  = self.ucb_param + delta
                self.explore_profit = 0
                self.explore_counter = 1



        return True





