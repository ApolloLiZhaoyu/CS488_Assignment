import math
import pickle
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.pyplot import cm
from utils.config import cfg
from utils.parse_args import parse_args


def cond_intensity(t, events, dim):
    exp_decay = [[math.exp(-cfg.W * (t - t_i)) for t_i in d_events] for d_events in events]
    return cfg.MU[dim] + np.sum([cfg.A[d][dim] * exp_decay[d][i] for d in range(cfg.Z) for i in range(len(events[d]))])


def simulation():
    events = [[] for _ in range(cfg.Z)]
    intensities = [[cfg.MU[d]] for d in range(cfg.Z)]
    s = 0
    cnt = 0
    while s < cfg.T and cnt < cfg.SEQ_MAX_LEN:
        lamb_bar = np.sum([cond_intensity(s, events, d) for d in range(cfg.Z)])
        u = np.random.rand()
        w = - math.log(u) / lamb_bar
        s = s + w
        if s > cfg.T:
            break
        D = np.random.rand()
        lamb = [cond_intensity(s, events, d) for d in range(cfg.Z)]
        if D * lamb_bar <= np.sum(lamb):
            d = 0
            while d < cfg.Z:
                if D * lamb_bar <= np.sum(lamb[:d + 1]):
                    break
                d += 1
            events[d].append(s)
            intensities[d].append(lamb[d])
            cnt += 1
    return events, intensities


def flat(lists):
    return np.sort(np.array([element for d_list in lists for element in d_list]))


def visualization(events, intensities=None):
    plt.xlim((0, math.ceil(flat(events)[-1])))
    if intensities is not None:
        plt.ylim((-2, math.ceil(flat(intensities)[-1])))
    plt.xlabel('Time')

    color = cm.rainbow(np.linspace(0, 1, cfg.Z))
    step = np.linspace(0, 2, cfg.Z + 1)
    for d, d_events in enumerate(events):
        plt.plot(d_events, np.zeros_like(d_events) - step[d],
                 color='white', marker='^', markersize=10, markeredgecolor=color[d],
                 label='Events in dimension {}'.format(d))
        if intensities is not None:
            plt.plot([0] + d_events, intensities[d], label=r'Intensity $\lambda$(t) in dimension {}'.format(d))

    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    args = parse_args('Simulation of Multi-dimensional Hawkes Process using Thinning Algorithm.')
    np.random.seed(cfg.RANDOM_SEED)

    if cfg.GEN_DATA:
        for i in range(cfg.SEQ_NUM):
            events, intensities = simulation()
            pickle.dump(events, open('{}/events_{}.pkl'.format(cfg.OUT_DIR, i), 'wb'))
            pickle.dump(intensities, open('{}/intensities_{}.pkl'.format(cfg.OUT_DIR, i), 'wb'))

    # events = pickle.load(open('{}/events_{}.pkl'.format(cfg.OUT_DIR, 0), 'rb'))
    # intensities = pickle.load(open('{}/intensities_{}.pkl'.format(cfg.OUT_DIR, 0), 'rb'))
    # visualization(events, intensities)
