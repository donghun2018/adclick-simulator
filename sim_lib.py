"""
Simulator library

Donghun Lee 2018
"""
import pickle

import numpy as np

from policy_loader import get_pols, get_puids


def load_auction_p(fname):
    """
    loads auction output from a file

    :param fname: usually, auction_???.p
    :return: auct. This can be fed to simulator through read_in_auction method
    """
    return pickle.load(open(fname, "rb"))


def load_policies(all_attrs, possible_bids, max_T):
    """
    loads policies from individual .py files. See policy_loader for more info

    :return: policy object list, policy-unique-id (puid) string list
    """
    pols = [pol(all_attrs, possible_bids, max_T) for pol in get_pols()]
    return pols, get_puids()


def get_click_prob(theta, bid):
    """
    click probability, given theta and bid, using logistic function

    :param theta: dict, needs keys 'a', 'bid', '0', and 'max_click_prob'
    :param bid: will be converted to float
    :return: probability of click
    """
    th = theta['a'] + theta['bid'] * float(bid) + theta['0']  # TODO: really? have bid in this section????
    p_click = theta['max_click_prob'] / (1 + np.exp(-th))  # TODO: introduce more robust function
    return p_click


def compute_second_price_cost(bids, size=1):
    """ returns second price cost. if all bids are the same, then cost is the bid.

    :param bids: list.
    :return:
    """
    ubids = sorted(list(set(bids)))
    if len(ubids) >= 2:
        return [ubids[-2]] * size
    else:
        return [ubids[0]] * size


def _compute_actual_second_price_cost(bid, sorted_unique_bid_list):
    """

    :param bid: bid price
    :param sorted_unique_bid_list: MUST BE SORTED AND UNIQUE (increasing order)
                                   must have bid as its element
    :return: second price cost (if lowest, then itself)
    """

    ix = sorted_unique_bid_list.index(bid)
    cost_ix = 0 if ix == 0 else ix - 1
    return sorted_unique_bid_list[cost_ix]



def max_ix(l):
    """ returns all indices of l whose element is the max value

    :param l: iterable
    :return: index list
    """
    max_l = max(l)
    ret = []
    for ix, item in enumerate(l):
        if item == max_l:
            ret.append(ix)
    return ret


def top_K_max(l, K=1, prng=None):
    """
    returns K elements such that no element in returned list is less than the largest element of l that are not returned
    :param l: list
    :param K: int
    :param prng: numpy-compatible PRNG, can be obtained from np.random.RandomState()
    :return: length-K list containing elements, and second return length-K list containing corresponding index in input l

    Note that ties are randomly broken by random.shuffle function from numpy
    """
    if prng is None:
        prng = np.random.RandomState()
    ret_ix = []
    l2 = sorted(list(set(l)), reverse=True)
    for v in l2:
        indices = []
        for ix, item in enumerate(l):
            if item == v:
                indices.append(ix)
        prng.shuffle(indices)
        ret_ix.extend(indices)
        if len(ret_ix) > K:
            break

    ret2 = ret_ix[:K]
    ret1 = [l[i] for i in ret2]

    return ret1, ret2


