import math
import time
import pickle
import numpy as np

from utils.config import cfg
from utils.parse_args import parse_args
from data.dataset import ATMDataset


def g(t):
    return math.exp(-cfg.W * t)

def G(t):
    return -(math.exp(-cfg.W * t) - 1) / cfg.W


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


def calc_A(samples, A, U1, U2, Z1, Z2, mu, p_list, g_diff_list):
    B = np.zeros((cfg.ADM4.U, cfg.ADM4.U))
    G_sum = np.zeros(cfg.ADM4.U)
    C = np.zeros((cfg.ADM4.U, cfg.ADM4.U))
    for events in samples:
        for j in range(len(events)):
            _u = events[j]['dim']
            G_sum[_u] += G(cfg.ADM4.T - events[j]['time'])

    for _u in range(cfg.ADM4.U):
        for idx, events in enumerate(samples):
            p = p_list[idx]
            g_diff = g_diff_list[idx]
            for i in range(len(events)):
                u = events[i]['dim']
                C[u][_u] += p[i][i] * A[u][_u] * g_diff[i][_u] / mu[u]

    for u in range(cfg.ADM4.U):
        for _u in range(cfg.ADM4.U):
            B[u][_u] = G_sum[_u] + cfg.ADM4.RHO * (-Z1[u][_u] + U1[u][_u] - Z2[u][_u] + U2[u][_u])
            A[u][_u] = (-B[u][_u] + math.sqrt(B[u][_u] * B[u][_u] + 8 * cfg.ADM4.RHO * C[u][_u])) / (4 * cfg.ADM4.RHO)

    return A


def calc_Z1(A, U1):
    U, sigma, VT = np.linalg.svd(A + U1, full_matrices=False)
    th = cfg.ADM4.LAMBDA1 / cfg.ADM4.RHO
    sigma = sigma - th
    sigma[sigma < 0] = 0
    sigma = np.diag(sigma)
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


def train(samples):
    A = np.random.rand(cfg.ADM4.U, cfg.ADM4.U)
    mu = np.random.rand(cfg.ADM4.U)
    U1 = np.zeros_like(A)
    U2 = np.zeros_like(A)
    # g_diff_list = calc_g_diff_list(samples)
    # print('Calculate g_diff complete!')
    # pickle.dump(g_diff_list, open('g_diff_list.pkl', 'wb'))
    g_diff_list = pickle.load(open('g_diff_list.pkl', 'rb'))
    for k in range(cfg.ADM4.ITER_NUM):
        start = time.time()
        print('Iteration', k)
        converge = False
        Z1 = calc_Z1(A, U1)
        Z2 = calc_Z2(A, U2)
        U1 = U1 + (A - Z1)
        U2 = U2 + (A - Z2)
        while not converge:
            _A = A.copy()
            p_list = calc_p_list(samples, A, mu, g_diff_list)
            # print('Finish calculating p.')
            A = calc_A(samples, A, U1, U2, Z1, Z2, mu, p_list, g_diff_list)
            # print('Finish calculating A.')
            mu = calc_mu(samples, p_list)
            # print('Finish calculating mu.')

            if np.linalg.norm(A - _A) < cfg.ADM4.TH:
                converge = True
                print(np.linalg.norm(A - _A))
            else:
                print(np.linalg.norm(A - _A))

        end = time.time()
        print("It takes {} s to run a iteration.".format(end - start))

    return A, mu


def cond_intensity(t, events, dim, A, mu):
    exp_decay = [[math.exp(-cfg.W * (t - t_i)) for t_i in d_events] for d_events in events]
    return mu[dim] + np.sum([A[d][dim] * exp_decay[d][i] for d in range(cfg.ADM4.U) for i in range(len(events[d]))])


def predict(A, mu, events):
    test_events = [[] for _ in range(cfg.ADM4.U)]
    test_events[events[0]['dim']].append(events[0]['time'])
    pred_events = []
    s = events[0]['time']
    cnt = 1
    while cnt < len(events):
        lamb_bar = np.sum([cond_intensity(s, test_events, d, A, mu) for d in range(cfg.ADM4.U)])
        u = np.random.rand()
        w = - math.log(u) / lamb_bar
        pred_s = s + w
        D = np.random.rand()
        lamb = [cond_intensity(pred_s, test_events, d, A, mu) for d in range(cfg.ADM4.U)]
        if D * lamb_bar <= np.sum(lamb):
            d = 0
            while d < cfg.ADM4.U:
                if D * lamb_bar <= np.sum(lamb[:d + 1]):
                    break
                d += 1
            event = dict()
            event['name'] = events[cnt]['name']
            event['dim'] = d
            event['time'] = pred_s
            pred_events.append(event)
            test_events[events[cnt]['dim']].append(events[cnt]['time'])
            s = events[cnt]['time']
            cnt += 1
    return pred_events


def test(samples, A, mu):
    pred_samples = []
    for events in samples:
        pred_events = predict(A, mu, events)
        pred_samples.append(pred_events)
    return pred_samples


def evaluate(pred_samples, samples):
    pred_cnt = np.zeros(cfg.ADM4.U)
    gt_cnt = np.zeros(cfg.ADM4.U)
    match_cnt = np.zeros(cfg.ADM4.U)
    MAE = 0
    cnt = 0

    for pred_events, events in zip(pred_samples, samples):
        events = events[1:]
        cnt += len(pred_events)
        for pred_event, event in zip(pred_events, events):
            pred_type = pred_event['dim']
            gt_type = event['dim']
            pred_cnt[pred_type] += 1
            gt_cnt[gt_type] += 1
            if pred_type == gt_type:
                match_cnt[pred_type] += 1
            MAE += abs(pred_event['time'] - event['time'])
    
    print(match_cnt)
    print(pred_cnt)
    print(gt_cnt)

    precision = match_cnt / pred_cnt
    recall = match_cnt / gt_cnt
    f1_score = 2 * precision * recall / (precision + recall)
    MAE /= cnt
    return precision, recall, f1_score, MAE


if __name__ == '__main__':
    args = parse_args('Use ADM4 algorithm to fit and predict on a ATM dataset.')
    np.random.seed(0)

    train_dataset = ATMDataset(mode='train')
    test_dataset = ATMDataset(mode='test')

    # A, mu = train(train_dataset)
    # pickle.dump(A, open('A.pkl', 'wb'))
    # pickle.dump(mu, open('mu.pkl', 'wb'))

    A = pickle.load(open('A.pkl', 'rb'))
    mu = pickle.load(open('mu.pkl', 'rb'))

    pred_samples = test(test_dataset, A, mu)
    print('Finish predicting samples.')
    precision, recall, f1_score, MAE = evaluate(pred_samples, test_dataset)
    print('Precision of the pred samples is {}, avg is {}'.format(precision, precision.mean()))
    print('Recall of the pred samples is {}, avg is {}'.format(recall, recall.mean()))
    print('F1_score of the pred samples is {}, avg is {}'.format(f1_score, f1_score.mean()))
    print('MAE of the pred samples is', MAE)

