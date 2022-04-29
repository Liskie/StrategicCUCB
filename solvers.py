from __future__ import division

from typing import List, NoReturn, Union

import numpy as np
import time
from scipy.stats import beta
from tqdm import tqdm

from bandits import BernoulliBandit, Bandit, CABandit


class Solver(object):
    def __init__(self, bandit):
        """
        bandit (Bandit): the target bandit to solve.
        """
        #        assert isinstance(bandit, BernoulliBandit)
        np.random.seed(int(time.time()))

        self.bandit = bandit

        self.counts = [0] * self.bandit.n
        self.actions = []  # A list of machine ids, 0 to bandit.n-1.
        self.regret = 0.  # Cumulative regret.
        self.regrets = [0.]  # History of cumulative regret.

    def update_regret(self, i):
        # i (int): index of the selected machine.
        self.regret += self.bandit.best_proba - self.bandit.probas[i]
        self.regrets.append(self.regret)

    @property
    def estimated_probas(self):
        raise NotImplementedError

    def run_one_step(self):
        """Return the machine index to take action on."""
        raise NotImplementedError

    def run(self, num_steps):
        assert self.bandit is not None
        for _ in range(num_steps):
            i = self.run_one_step()

            self.counts[i] += 1
            self.actions.append(i)
            self.update_regret(i)


class UCB1(Solver):
    def __init__(self, bandit, init_proba=1.0):
        super(UCB1, self).__init__(bandit)
        self.t = 0
        self.estimates = [init_proba] * self.bandit.n

    @property
    def estimated_probas(self):
        return self.estimates

    def run_one_step(self):
        self.t += 1

        # Pick the best one with consideration of upper confidence bounds.
        i = max(range(self.bandit.n), key=lambda x: self.estimates[x] + np.sqrt(
            2 * np.log(self.t) / (1 + self.counts[x])))
        r = self.bandit.generate_reward(i)

        self.estimates[i] += 1. / (self.counts[i] + 1) * (r - self.estimates[i])

        return i


class CACUCB:
    '''
    Solver designer for crowdsourcing annotation
    '''

    def __init__(self, bandit: CABandit, ucb_scale: float = 1.0):
        self.bandit = bandit

        self.t = 0  # Time step
        self.ucb_scale = ucb_scale

        self.cumulative_regret = 0
        self.regret_history = []

        self.action_history = []

    def update_regret(self, ids: Union[List[int], int]) -> NoReturn:
        '''
        Update cumulative regret by selected super-arm.
        Args:
            ids: List of arm ids or an arm id in the selected super-arm.
        '''
        best_expected_reward = sum(self.bandit.arm_best_win_probs.values())
        if isinstance(ids, list):
            current_expected_reward = sum(self.bandit.arm_true_win_probs[id] for id in ids)
        elif isinstance(ids, int):
            current_expected_reward = self.bandit.arm_true_win_probs[ids]
        else:
            raise TypeError('Input is neither integer id nor list of ids.')
        self.cumulative_regret += best_expected_reward - current_expected_reward
        self.regret_history.append(self.cumulative_regret)

    def initialize(self) -> NoReturn:
        '''
        Before normal rounds, play each arm once.
        '''
        for id in tqdm(self.bandit.arm_ids, desc='Initializing'):
            self.t += 1
            self.bandit.arm_pred_win_probs[id] = self.bandit.generate_reward(id)
            self.bandit.arm_play_counters[id] += 1
            self.action_history.append(id)
            self.update_regret(id)

    def run_one_step(self) -> NoReturn:
        '''
        Pick the best super-arm according to current knowledge and play it.
        '''
        self.t += 1

        # Pick the best super-arm with consideration of upper confidence bounds.
        arm_pred_win_probs_with_ucb = {id: self.bandit.arm_pred_win_probs[id] +  # Predicted outcome of arm i so far
                                           self.ucb_scale * np.sqrt(3 * np.log(self.t)
                                                                    / (2 * self.bandit.arm_play_counters[id]))  # ucb
                                       for id in self.bandit.arm_ids}

        picked_arm_ids = [item[0] for item in sorted(arm_pred_win_probs_with_ucb.items(),
                                                     key=lambda item: item[1],
                                                     reverse=True)[:self.bandit.super_arm_size]]

        # Play them.
        for id in picked_arm_ids:
            reward = self.bandit.generate_reward(id)
            # Update predicted winning rate with current average of the arm total outcome
            self.bandit.arm_pred_win_probs[id] = (self.bandit.arm_play_counters[id]
                                                  * self.bandit.arm_pred_win_probs[id] + reward) \
                                                 / (self.bandit.arm_play_counters[id] + 1)
            self.bandit.arm_play_counters[id] += 1

        self.action_history.append(picked_arm_ids)

        self.update_regret(picked_arm_ids)

    def run(self, num_steps: int) -> NoReturn:
        '''
        Run the whole process.
        Args:
            num_steps: Number of time steps to take.
        '''
        assert self.bandit is not None
        assert num_steps >= self.bandit.arm_num, \
            f'Given num_steps {num_steps} are less than arm_num {self.bandit.arm_num}.'
        self.initialize()
        for _ in tqdm(range(num_steps - self.bandit.arm_num), desc='Running'):
            self.run_one_step()


class CUCB(Solver):
    def __init__(self, bandit, init_proba=1.0, Bmax=0, card=2, scale=0.1):
        super(CUCB, self).__init__(bandit)
        self.t = 0
        self.estimates = np.zeros(self.bandit.n)
        self.counts = np.zeros(self.bandit.n)
        self.Bmax = Bmax
        self.card = card
        self.scale = scale
        self.best_combo = self.bandit.combinatorial_best(card)
        self.pulls = [i for i in range(self.bandit.n)]
        # print(self.estimates)
        # print(self.counts)

    @property
    def estimated_probas(self):
        return self.estimates

    def run_one_step(self):
        self.t += 1

        # Pick the best one with consideration of upper confidence bounds.
        UCB_i = [self.estimates[x] +
                 self.scale * np.sqrt(3 * np.log(self.t) / (2 * self.counts[x])) +
                 self.scale * (self.Bmax / self.counts[x])
                 for x in range(self.bandit.n)]  # [0]
        UCB_i = (-np.array(UCB_i)).argsort()[:self.card]
        played = []
        if UCB_i[0] not in self.bandit.best_arms and UCB_i[1] not in self.bandit.best_arms:
            self.pulls.append(self.pulls[-1] + 1)
        else:
            self.pulls.append(self.pulls[-1])
        # play the best 5 arms
        for i in range(self.card):
            r = self.bandit.generate_reward(UCB_i[i])
            self.estimates[UCB_i[i]] = (self.counts[UCB_i[i]] * self.estimates[UCB_i[i]] + r) / (
                    self.counts[UCB_i[i]] + 1)
            self.counts[UCB_i[i]] += 1
            played.append(UCB_i[i])
        # print(played)
        # print(self.estimates)
        return played

    def initialize(self, i):
        self.t += 1
        other = np.random.randint(self.bandit.n - 1, size=1)
        if other[0] == i:
            other = np.random.randint(self.bandit.n - 1, size=1)
        # print(other[0])
        r = self.bandit.generate_reward(other[0])
        self.estimates[other[0]] = (self.counts[other[0]] * self.estimates[other[0]] + r) / (self.counts[other[0]] + 1)
        self.counts[other[0]] += 1

        r = self.bandit.generate_reward(i)
        self.estimates[i] = (self.counts[i] * self.estimates[i] + r) / (self.counts[i] + 1)
        self.counts[i] += 1

        return [other[0], i]

    def run(self, num_steps):
        assert self.bandit is not None
        for t in range(self.bandit.n):
            played = self.initialize(t)
            self.actions.append(played)
            self.update_regret(played)
        # print(self.estimates)
        for _ in range(num_steps - self.bandit.n):
            played = self.run_one_step()

            for i in played:
                self.actions.append(i)

            self.update_regret(played)

    def update_regret(self, arr_i):
        # i (int): index of the selected machine.
        total_prob = 0
        for i in arr_i:
            total_prob += self.bandit.probas[i]

        self.regret += self.best_combo - total_prob
        self.regrets.append(self.regret)

    def clear(self):
        self.regret = 0
        self.regrets = []
        self.estimates = np.zeros(self.bandit.n)
        self.counts = np.zeros(self.bandit.n)


class naive_CUCB(Solver):
    def __init__(self, bandit, init_proba=1.0, Bmax=0, card=2, scale=0.1):
        super(naive_CUCB, self).__init__(bandit)
        self.t = 0  # time step
        self.estimates = np.zeros(self.bandit.n)
        self.counts = np.zeros(self.bandit.n)
        self.Bmax = Bmax
        self.card = card
        self.scale = scale
        self.best_combo = self.bandit.combinatorial_best(card)
        self.pulls = [i for i in range(self.bandit.n)]
        # print(self.estimates)
        # print(self.counts)

    @property
    def estimated_probas(self):
        return self.estimates

    def run_one_step(self):
        self.t += 1

        # Pick the best one with consideration of upper confidence bounds.
        UCB_i = [self.estimates[x] +  # Mean of all outcomes of arm i so far
                 self.scale * np.sqrt(3 * np.log(self.t) / (2 * self.counts[x]))  # ucb, scale = 1 by default
                 for x in range(self.bandit.n)]

        UCB_i = (-np.array(UCB_i)).argsort()[:self.card]  # the best arm index when card = 1

        if UCB_i[0] not in self.bandit.best_arms and UCB_i[1] not in self.bandit.best_arms:
            self.pulls.append(self.pulls[-1] + 1)
        else:
            self.pulls.append(self.pulls[-1])

        played = []
        # play the best card num arms
        for i in range(self.card):
            r = self.bandit.generate_reward(UCB_i[i])  # UCB_i[i} is the best arm index
            self.estimates[UCB_i[i]] = (self.counts[UCB_i[i]] * self.estimates[UCB_i[i]] + r) \
                                       / (self.counts[UCB_i[i]] + 1)  # New mean of the arm total output
            self.counts[UCB_i[i]] += 1
            played.append(UCB_i[i])
        # print(played)
        # print(self.estimates)
        return played

    def initialize(self, i):
        self.t += 1
        r = self.bandit.generate_reward(i)
        self.estimates[i] = (self.counts[i] * self.estimates[i] + r) / (self.counts[i] + 1)
        self.counts[i] += 1

    def run(self, num_steps):
        assert self.bandit is not None
        for t in range(self.bandit.n):
            self.initialize(t)
            self.actions.append(t)
            self.update_regret([t])

        for _ in range(num_steps - self.bandit.n):
            played = self.run_one_step()

            for i in played:
                self.actions.append(i)

            self.update_regret(played)

    def update_regret(self, arr_i):
        # i (int): index of the selected machine.
        total_prob = 0
        for i in arr_i:
            total_prob += self.bandit.probas[i]

        self.regret += self.best_combo - total_prob
        self.regrets.append(self.regret)

    def clear(self):
        self.regret = 0
        self.regrets = []
        self.estimates = np.zeros(self.bandit.n)
        self.counts = np.zeros(self.bandit.n)
