import math
import pickle
import numpy as np

from utils.config import cfg
from data.dataset import ATMDataset


def g(t):
    return cfg.W * math.exp(-cfg.W * t)


def calc_g_diff(events):
    g_diff = np.zeros((len(events), cfg.ADM4.U))
    for i in range(len(events)):
        time = events[i]['time']
        for j in range(i):
            _u = events[j]['dim']
            g_diff[i][_u] += g(time - events[j]['time'])
    return g_diff


def calc_g_diff_list(samples):
    g_diff_list = []
    for idx, events in enumerate(samples):
        print(idx, len(events))
        g_diff_list.append(calc_g_diff(events))
    return g_diff_list


def calc_p(events, A, mu, g_diff):
    n_c = g_diff.shape[0]
    p = np.zeros((n_c, n_c))
    print(p.shape)
    for i in range(n_c):
        u = events[i]['dim']
        time = events[i]['time']
        numerator = mu[u]
        denominator = mu[u] + A[u].dot(g_diff[i])
        p[i][i] = numerator / denominator
        for j in range(i):
            _u = events[j]['dim']
            p[i][j] = p[i][i] * A[u][_u] * g(time - events[j]['time']) / mu[u]
    return p


def calc_p_list(samples, A, mu, g_diff_list):
    p_list = []
    for i in range(len(samples)):
        p_list.append(calc_p(samples[i], A, mu, g_diff_list[i]))
    return p_list


def calc_mu(samples, p_list):
    numerator = np.zeros(cfg.ADM4.U)
    denominator = 0
    for idx, events in enumerate(samples):
        p = p_list[idx]
        for i in range(len(events)):
            u = events[i]['dim']
            numerator[u] += p[i][i]
        denominator += events[-1]['time']
    return numerator / denominator


def calc_A(samples, A, U1, U2, Z1, Z2, p_list):
    B = np.zeros((cfg.ADM4.U, cfg.ADM4.U))
    G_sum = np.zeros(cfg.ADM4.U)
    C = np.zeros((cfg.ADM4.U, cfg.ADM4.U))
    for events in samples:
        for j in range(len(events)):
            _u = events[j]['dim']
            G_sum[_u] += g(cfg.ADM4.T - events[j]['time']) - g(0)

    for idx, events in enumerate(samples):
        p = p_list[idx]
        for i in range(len(events)):
            for j in range(i):
                C[i][j] += p[i][j]

    for u in range(cfg.ADM4.U):
        for _u in range(cfg.ADM4.U):
            B[u][_u] += G_sum[_u] + cfg.ADM4.RHO * (-Z1[u][_u] + U1[u][_u] - Z2[u][_u] + U2[u][_u])
            A[u][_u] = (-B[u][_u] + math.sqrt(B * B + 8 * cfg.ADM4.RHO * C[u][_u])) / (4 * cfg.ADM4.RHO)

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
    # g_diff_list = calc_g_diff_list(samples)
    # print('Calculate g_diff complete!')
    # f = open('g_diff.pkl', 'wb')
    # pickle.dump(g_diff_list, f)
    f = open('g_diff.pkl', 'rb')
    g_diff_list = pickle.load(f)
    f.close()
    converge = False
    for k in range(cfg.ADM4.ITER_NUM):
        Z1 = calc_Z1(A, U1)
        Z2 = calc_Z2(A, U2)
        U1 = U1 + (A - Z1)
        U2 = U2 + (A - Z2)
        while not converge:
            _A = A.copy()

            p_list = calc_p_list(samples, A, mu, g_diff_list)
            A = calc_A(samples, A, U1, U2, Z1, Z2, p_list)
            mu = calc_mu(samples, p_list)

            if np.linalg.norm(A - _A) < cfg.ADM4.TH:
                converge = True
                print(np.linalg.norm(A - _A))
            else:
                print(np.linalg.norm(A - _A))

    return A, mu


if __name__ == '__main__':
    atm_dataset = ATMDataset(mode='train')
    A, mu = ADM4(atm_dataset)
