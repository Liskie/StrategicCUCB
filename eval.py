import matplotlib  # noqa

# matplotlib.use('Agg')  # noqa

import matplotlib.pyplot as plt
import numpy as np

from bandits import BernoulliBandit, CABandit
from solvers import Solver, UCB1, CUCB, naive_CUCB, CACUCB
from utils import get_worker_ids, gen_train_data_with_picked_super_arm


def plot_results(solvers, solver_names, figname):
    """
    Plot the results by multi-armed bandit solvers.

    Args:
        solvers (list<Solver>): All of them should have been fitted.
        solver_names (list<str)
        figname (str)
    
    assert len(solvers) == len(solver_names)
    assert all(map(lambda s: isinstance(s, Solver), solvers))
    assert all(map(lambda s: len(s.regrets) > 0, solvers))
    """

    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(bottom=0.15, wspace=0.3)
    ax1 = fig.add_subplot(111)

    # Sub.fig. 1: Regrets in time.
    line_style = ["--", "-", '-', '-', '-']
    markers = ['^', 'o', '+', '*', 's']
    color = ['red', 'b', 'tomato']
    reg_list = []
    B_list = [i for i in range(0, 102, 2)]
    for i, s in enumerate(solvers):
        reg_list.append(s.regrets[-1])
        ax1.plot(range(len(s.regrets)), s.regrets, label=solver_names[i], linewidth=2, linestyle=line_style[i])

    ax1.set_xlabel('Time horizon', fontsize=18)
    ax1.set_ylabel('Cumulative regret', fontsize=18)
    plt.tick_params(labelsize=16)

    plt.title("Combinatorial Strategic - UCB", fontsize=20)
    ax1.legend(fontsize=15, loc="upper left")
    # ax1.legend(loc=9, bbox_to_anchor=(1.82, -0.25), ncol=5)
    ax1.grid('k', ls='--', alpha=0.3)
    plt.savefig(figname)


def experiment(K, N, card=3):
    """
    Run a small experiment on solving a Bernoulli bandit with K slot machines,
    each with a randomly initialized reward probability.

    Args:
        K (int): number of slot machines.
        N (int): number of time steps to try.
    """
    prob = [0.1, 0.2, 0.3, 0.4, 0.45, 0.56, 0.6, 0.55, 0.9, 0.95]
    b = BernoulliBandit(K)
    print("Randomly generated Bernoulli bandit has reward probabilities:\n"), b.probas

    # optimal = max(range(K), key=lambda i: b.probas[i])

    # randomly assign budget, maximum budget = 10
    budget = [10 for i in range(K)]  # np.random.randint(100, size=K)
    best_arms = (-np.array(b.probas)).argsort()[:2]

    for i in best_arms:
        budget[i] = 0

    b.set_budget(budget)

    # ======================== b1 ======================== #
    b1 = BernoulliBandit(K, LSI=True)
    budget1 = [0 for i in range(K)]  # np.random.randint(50, size=K)
    best_arms1 = (-np.array(b1.probas)).argsort()[:2]
    for i in best_arms1:
        budget1[i] = 0

    b1.set_budget(budget1)

    # ======================== b2 ================================== #
    b2 = BernoulliBandit(K, LSI=True)
    budget2 = [10 for i in range(K)]
    best_arms2 = (-np.array(b2.probas)).argsort()[:2]
    for i in best_arms2:
        budget2[i] = 0

    b2.set_budget(budget2)

    # ======================== b3 ============================ #
    b3 = BernoulliBandit(K)
    budget3 = [50 for i in range(K)]
    best_arms3 = (-np.array(b3.probas)).argsort()[:2]
    for i in best_arms3:
        budget3[i] = 0

    b3.set_budget(budget3)

    # ======================== b4 =========================== #
    b4 = BernoulliBandit(K, LSI=True)
    budget4 = [100 for i in range(K)]
    best_arms4 = (-np.array(b4.probas)).argsort()[:2]
    for i in best_arms4:
        budget4[i] = 0

    b4.set_budget(budget4)

    b_list = []
    test_solvers = [
        naive_CUCB(b, Bmax=10, scale=0.5, card=2),
        CUCB(b1, Bmax=0, scale=0.2, card=2),
        CUCB(b2, Bmax=10, scale=0.2, card=2),
        CUCB(b3, Bmax=50, scale=0.1, card=2),
        CUCB(b4, Bmax=100, scale=0.1, card=2),
    ]

    names = [
        'Naive CUCB - Bmax=10',
        'Strategic CUCB - Bmax=0',  # 'Maximum Budget = 2',
        'Strategic CUCB - Bmax=10',
        'Strategic CUCB - Bmax=50',
        'Strategic CUCB - Bmax=100'
    ]

    for s in test_solvers:
        regret = [0 for i in range(N)]
        # average over 10 trials
        for i in range(10):
            s.clear()
            s.run(N)
            regret = np.add(regret, s.regrets[:N])
            s.clear()
        s.regrets = np.divide(regret, 10)

    plot_results(test_solvers, names, "results_K{}_N{}.png".format(K, N))


def run_CACUCB(num_step: int = 100):
    '''
    Run CUCB on the crowdsourcing annotation dataset.
    Select the best super-arm.
    '''

    worker_ids = get_worker_ids()

    bandit = CABandit(worker_ids)
    solver = CACUCB(bandit)
    solver.run(num_step)
    plt.plot(range(num_step), solver.regret_history)
    plt.show()

    print('Best super-arm:   ', sorted(list(bandit.arm_best_win_probs.keys())))
    picked_super_arm = sorted(list(solver.action_history[-1]))
    print('Picked super-arm: ', picked_super_arm)
    gen_train_data_with_picked_super_arm(picked_super_arm)

    with open('action_log.txt', 'w') as out:
        for action in solver.action_history:
            out.write(f'{action}\n')




if __name__ == '__main__':
    # experiment(10, 5000)
    run_CACUCB(5000)
