import argparse

from utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir


def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--cfg', dest='cfg_file', action='append',
                        help='an optional config file', default=None, type=str)
    args = parser.parse_args()

    # load cfg from file
    if args.cfg_file is not None:
        for f in args.cfg_file:
            cfg_from_file(f)

    out_dir = get_output_dir()
    cfg_from_list(['OUT_DIR', out_dir])

    return args
