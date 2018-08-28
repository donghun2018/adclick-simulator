"""
Loads policies by
1. load the list of PUID from "puid_list.csv"
2. load "Policy_<PUID>" class that is defined in ./Policies/<PUID>.py

by Donghun Lee 2018
"""

import csv
import os
from importlib import import_module
from itertools import repeat

def get_pols(puids=None):
    """
    dynamically load policy functions from subpackage Policies

    :param puids: name of .py files to be loaded (if None, "puid_list.csv" is used)
    :return:
    """
    if not puids:
        puids = get_puids()
    mod_names = [".Policies" + "." + puid for puid in puids]
    pol_names = ["Policy_" + puid for puid in puids]
    mods = list(map(import_module, mod_names, repeat("Simulator")))
    pol_ptr = [getattr(mod, pol) for mod, pol in zip(mods, pol_names)]
    return pol_ptr


def get_puids():
    """
    Use puid_list.csv to get names of the policies to be used
    :return:
    """
    with open(os.path.dirname(__file__) + os.sep + "puid_list.csv") as ifh:
        reader = csv.reader(ifh)
        puids = [fn[0] for fn in reader]
    return puids


if __name__ == "__main__":
    # this is how you can use the policy loader in other files -- DH
    #from policy_loader import get_pols
    pols = get_pols()
    for pol in pols:
        policy = pol()
        print("policy name = " + policy.id())
        print("policy_bidspace = " + str(policy.bid_space))
        print("a sample bid : " + str(policy.bid()))
