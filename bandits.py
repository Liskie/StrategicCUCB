from __future__ import division

import time
from typing import List

import numpy as np
from tqdm import tqdm

from utils import get_super_arm_f1_by_id


class Bandit(object):

    def generate_reward(self, i):
        raise NotImplementedError


class BernoulliBandit(Bandit):
    def __init__(self, n, probas=None, budget=None, LSI=True):
        assert probas is None or len(probas) == n
        self.n = n  # number of arms(crowd workers)
        self.B = [0 for i in range(n)]  # budget
        self.count = [0 for i in range(n)]
        self.LSI = LSI
        if probas is None:
            np.random.seed(int(time.time()))
            self.probas = [np.random.random() for _ in range(self.n)]
        else:
            self.probas = probas

        self.best_proba = max(self.probas)
        self.best_arms = (-np.array(self.probas)).argsort()[:2]

    def set_budget(self, budget):
        self.B = budget

    def generate_reward(self, i):
        # The player selected the i-th machine.
        if np.random.random() < self.probas[i]:
            reward = 1
        else:
            reward = 0

        if self.LSI:
            if self.count[i] == 0:
                self.count[i] += 1
                return reward + self.B[i]
            else:
                self.count[i] += 1
                return reward
        else:  # Non-LSI strategy
            if self.B[i] > 0:
                strategy = np.random.random()
                if self.B[i] - strategy < 0:
                    return reward
                else:
                    self.B[i] -= strategy
                    return reward + strategy
            else:
                return reward

    def combinatorial_best(self, card=2):
        best_arms = (-np.array(self.probas)).argsort()[:card]
        print("best arms", best_arms)
        total_prob = 0
        for a in best_arms:
            total_prob += self.probas[a]
        print(self.probas)
        return total_prob


class CABandit(Bandit):
    '''
    Bandit designed for the crowdsourcing annotation dataset.
    '''

    def __init__(self, worker_ids: List[int], super_arm_size: int = 20):
        self.arm_ids = worker_ids
        self.super_arm_size = super_arm_size
        self.arm_num = len(self.arm_ids)
        # How many times each arm is played.
        self.arm_play_counters = {arm_id: 0 for arm_id in self.arm_ids}
        # The actual winning rate of each arm.
        self.arm_true_win_probs = {arm_id: get_super_arm_f1_by_id(arm_id)
                                   for arm_id in tqdm(self.arm_ids, desc='Reading data')}
        self.arm_best_win_probs = dict(sorted(self.arm_true_win_probs.items(),
                                              key=lambda item: item[1],
                                              reverse=True)[:self.super_arm_size])
        # The calculated winning rate of each arm from trials so far.
        self.arm_pred_win_probs = {}
        # self.best_win_prob = max(self.arm_true_win_probs.values())

    def generate_reward(self, id: int) -> int:
        return 1 if np.random.random() < self.arm_true_win_probs[id] else 0


# To evaluate collusion strategy
class other_Bandit(Bandit):

    def __init__(self, n, probas=None, budget=None, ptype=''):
        assert probas is None or len(probas) == n
        self.n = n
        self.B = [0 for i in range(n)]
        self.count = [0 for i in range(n)]
        if probas is None:
            np.random.seed(int(time.time()))
            self.probas = [np.random.random() for _ in range(self.n)]
        else:
            self.probas = probas

        self.best_proba = max(self.probas)
        self.best_arms = (-np.array(self.probas)).argsort()[:2]
        self.ptype = ptype

    def set_budget(self, budget):
        self.B = budget
        if self.ptype == 'B':
            self.priority = (np.argsort(self.B)[::-1]).tolist()
        elif self.ptype == 'delta':
            self.priority = np.argsort(self.probas).tolist()
        elif self.ptype == 'Bdelta':
            self.priority = (np.argsort(np.add(self.B, self.probas))[::-1]).tolist()

    def generate_reward(self, i):
        # The player selected the i-th machine.
        if np.random.random() < self.probas[i]:
            reward = 1
        else:
            reward = 0

        # LSI strategy
        if self.count[i] == self.priority.index(i) + 1:
            self.count[i] += 1
            return reward + self.B[i]
        elif self.count[i] == 1 and self.B[i] > self.best_proba - self.probas[i]:
            self.count[i] += 1
            self.B[i] -= self.best_proba - self.probas[i]
            return self.best_proba
        else:
            self.count[i] += 1
            return reward
        #        return reward + strategy

    def combinatorial_best(self, card=5):
        best_arms = (-np.array(self.probas)).argsort()[:2]
        print("best arms", best_arms)
        total_prob = 0
        for a in best_arms:
            total_prob += self.probas[a]
        print(self.probas)
        return total_prob
