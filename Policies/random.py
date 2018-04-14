import numpy as np

from .policy import Policy


class Policy_random(Policy):

    def __init__(self, all_attrs, possible_bids=range(10), max_t=10, randseed=123456789):
        super().__init__(all_attrs, possible_bids, max_t, randseed=randseed)
