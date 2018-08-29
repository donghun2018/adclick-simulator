import numpy as np

from .policy import Policy


class Policy_random(Policy):

    def __init__(self, sim_param, policy_param=None):
        super().__init__(sim_param, policy_param)
