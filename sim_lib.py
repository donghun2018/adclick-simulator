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