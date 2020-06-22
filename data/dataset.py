import os.path as osp
import csv
from collections import defaultdict

from utils.config import cfg


class ATMDataset(object):
    def __init__(self, mode='train'):
        if mode == 'train':
            self.dataset_name = cfg.TRAIN_DATASET_NAME
        else:
            self.dataset_name = cfg.TEST_DATASET_NAME

        self.dataset = []
        self.atm_name = defaultdict(int)
        self.atm_idx = 1
        self.min_time = cfg.ADM4.MIN_TIME

        dataset_reader = csv.reader(open(osp.abspath(osp.join(cfg.DATA_DIR, self.dataset_name))))
        next(dataset_reader)

        for data in dataset_reader:
            d = dict()
            d['name'] = data[0]
            d['time'] = float(data[1]) - self.min_time
            d['dim'] = int(data[2])

            if self.atm_name[d['name']] == 0:
                self.atm_name[d['name']] = self.atm_idx
                self.dataset.append([])
                self.atm_idx += 1

            self.dataset[self.atm_name[d['name']] - 1].append(d)

        if mode == 'train':
            for data in self.dataset:
                if len(data) > cfg.ADM4.MAX_EVENTS:
                    self.dataset.remove(data)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __iter__(self):
        return iter(self.dataset)


if __name__ == '__main__':
    atm = ATMDataset(mode='train')
    # print(len(atm))
    # print(atm[0])
    # for data in atm:
    #     print(len(data))



