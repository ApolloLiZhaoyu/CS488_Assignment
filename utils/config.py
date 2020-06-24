import os
import os.path as osp
import numpy as np

from easydict import EasyDict as edict

__C = edict()

cfg = __C

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..'))

# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

# Output directory
__C.OUT_DIR = ''

# Place outputs under an experiments directory
__C.EXP_DIR = ''

# Experiment name
__C.EXP_NAME = ''

# Random seed
__C.RANDOM_SEED = 2020

# Parameters
__C.Z = 10
__C.T = 100
__C.W = 0.01

__C.MU = np.array(
    [0, 0.001, 0.05, 0.1, 0.025, 0.01, 0.007, 0.03, 0.008, 0.004]
)

__C.A = np.array([
    [0.1, 0.07, 0.004, 0, 0.003, 0, 0.09, 0, 0.07, 0.025],
    [0, 0.05, 0.028, 0, 0.027, 0.065, 0, 0, 0.097, 0],
    [0.09, 0, 0.006, 0.045, 0, 0, 0.053, 0.01, 0, 0.083],
    [0.02, 0.03, 0, 0.073, 0.058, 0, 0.026, 0, 0, 0],
    [0.05, 0.09, 0, 0, 0.066, 0, 0, 0.033, 0.006, 0],
    [0.07, 0, 0, 0, 0, 0.075, 0.063, 0.078, 0.085, 0.095],
    [0, 0.02, 0.001, 0, 0.057, 0.091, 0.009, 0.065, 0, 0.073],
    [0, 0.09, 0, 0.088, 0, 0.078, 0, 0.09, 0.068, 0],
    [0, 0, 0.093, 0, 0.033, 0, 0.069, 0, 0.082, 0.033],
    [0.001, 0, 0.089, 0, 0.008, 0, 0.007, 0, 0, 0.052]
])

__C.SEQ_NUM = 1000
__C.SEQ_MAX_LEN = 200
__C.GEN_DATA = False

__C.ADM4 = edict()
__C.ADM4.ITER_NUM = 5
__C.ADM4.TH = 0.01
__C.ADM4.LAMBDA1 = 0.02
__C.ADM4.LAMBDA2 = 0.6
__C.ADM4.RHO = 0.1
__C.ADM4.U = 7
__C.ADM4.MIN_TIME = 16313
__C.ADM4.MAX_EVENTS = 10000
__C.ADM4.T = 215

__C.TRAIN_DATASET_NAME = 'atm_train.csv'
__C.TEST_DATASET_NAME = 'atm_test.csv'


def get_output_dir():
    """
    Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    """
    __C.EXP_DIR = 'len{}_num{}'.format(__C.SEQ_MAX_LEN, __C.SEQ_NUM)
    out_dir = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return out_dir


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        if type(b[k]) is not type(v):
            raise ValueError(('Type mismatch ({} vs. {}) '
                              'for config key: {}').format(type(b[k]),
                                                           type(v), k))
        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d.keys()
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d.keys()
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value
