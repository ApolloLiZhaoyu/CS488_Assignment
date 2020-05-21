import math
import numpy as np

from utils.config import cfg
from data.dataset import ATMDataset


def g(t):
    return cfg.W * math.exp(-cfg.W * t)


def calc_p(events, A, mu):
    n_c = len(events)
    p = np.zeros((n_c, n_c))
    print(p.shape)
    for i in range(n_c):
        numerator = mu[events[i]['dim']]
        denominator = mu[events[i]['dim']] + np.sum([A[events[i]['dim']][events[k]['dim']]
                                                     * g(events[i]['time'] - events[k]['time']) for k in range(i)])
        p[i][i] = numerator / denominator
        for j in range(i):
            numerator = A[events[i]['dim']][events[j]['dim']] * g(events[i]['time'] - events[j]['time'])
            denominator = mu[events[i]['dim']] + np.sum([A[events[i]['dim']][events[k]['dim']]
                                                        * g(events[i]['time'] - events[k]['time']) for k in range(i)])
            p[i][j] = numerator / denominator

    return p


def calc_p_list(samples, A, mu):
    p_list = []
    for events in samples:
        p_list.append(calc_p(events, A, mu))
    return p_list


def calc_mu(samples, mu, p_list):
    for u in range(cfg.ADM4.U):
        numerator = 0
        denominator = 0
        for idx, events in enumerate(samples):
            p = p_list[idx]
            for i in range(len(events)):
                if events[i]['dim'] == u:
                    numerator += p[i][i]
            denominator += events[-1]['time']
        mu[u] = numerator / denominator

    return mu


def calc_A(samples, A, U1, U2, Z1, Z2, p_list):
    for u in range(cfg.ADM4.U):
        for _u in range(cfg.ADM4.U):
            print(u, _u)
            B = 0
            C = 0
            for idx, events in enumerate(samples):
                for j in range(len(events)):
                    if events[j]['dim'] == _u:
                        B += g(cfg.ADM4.T - events[j]['time'])

                p = p_list[idx]

                for i in range(len(events)):
                    _u_list = []
                    if events[i]['dim'] == u:
                        for j in _u_list:
                            C += p[i][j]
                    if events[i]['dim'] == _u:
                        _u_list.append(i)

            B += cfg.ADM4.RHO * (-Z1[u][_u] + U1[u][_u] - Z2[u][_u] + U2[u][_u])
            A[u][_u] = (-B + math.sqrt(B * B + 8 * cfg.ADM4.RHO * C)) / (4 * cfg.ADM4.RHO)

    return A


def calc_Z1(A, U1):
    U, sigma, VT = np.linalg.svd(A + U1)
    th = cfg.ADM4.LAMBDA1 / cfg.ADM4.RHO
    sigma = sigma - th
    sigma[sigma < 0] = 0
    return U.dot(sigma).dot(VT)


def calc_Z2(A, U2):
    res = A + U2
    for i in range(cfg.ADM4.U):
        for j in range(cfg.ADM4.U):
            if res[i][j] >= cfg.ADM4.LAMBDA2 / cfg.ADM4.RHO:
                res[i][j] -= cfg.ADM4.LAMBDA2 / cfg.ADM4.RHO
            elif res[i][j] <= - cfg.ADM4.LAMBDA2 / cfg.ADM4.RHO:
                res[i][j] += cfg.ADM4.LAMBDA2 / cfg.ADM4.RHO
            else:
                res[i][j] = 0
    return res


def ADM4(samples):
    A = np.random.rand(cfg.ADM4.U, cfg.ADM4.U)
    mu = np.random.rand(cfg.ADM4.U)
    U1 = np.zeros_like(A)
    U2 = np.zeros_like(A)
    converge = False
    for k in range(cfg.ADM4.ITER_NUM):
        Z1 = calc_Z1(A, U1)
        Z2 = calc_Z2(A, U2)
        U1 = U1 + (A - Z1)
        U2 = U2 + (A - Z2)
        while not converge:
            _A = A.copy()

            p_list = calc_p_list(samples, A, mu)
            A = calc_A(samples, A, U1, U2, Z1, Z2, p_list)
            mu = calc_mu(samples, mu, p_list)

            if np.linalg.norm(A - _A) < cfg.ADM4.TH:
                converge = True
                print(np.linalg.norm(A - _A))
            else:
                print(np.linalg.norm(A - _A))

    return A, mu


if __name__ == '__main__':
    atm_dataset = ATMDataset(mode='train')
    A, mu = ADM4(atm_dataset.dataset)
