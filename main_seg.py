import sys, os

# Fix PYTHONPATH - ensure project root is first
# 清除 hpenext 路径冲突
sys.path = [p for p in sys.path if 'hpenext' not in p]

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root in sys.path:
    sys.path.remove(_project_root)
sys.path.insert(0, _project_root)

from tools.runner_seg import run_net, test_net
from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *
import time
import torch
from tensorboardX import SummaryWriter

def main():
    args = parser.get_args()
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True

    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True
        dist_utils.init_dist(args.launcher)
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)

    if not args.test:
        if args.local_rank == 0:
            train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
            val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))
        else:
            train_writer = None
            val_writer = None

    config = get_config(args, logger=logger)

    if args.distributed:
        assert config.total_bs % world_size == 0
        config.dataset.train.others.bs = config.total_bs // world_size
        if hasattr(config.dataset, 'val'):
            config.dataset.val.others.bs = config.total_bs // world_size * 2
    else:
        if hasattr(config.dataset, 'train'):
            config.dataset.train.others.bs = config.total_bs
        if hasattr(config.dataset, 'val'):
            config.dataset.val.others.bs = config.total_bs * 2

    log_args_to_file(args, 'args', logger=logger)
    log_config_to_file(config, 'config', logger=logger)

    if args.seed is not None:
        misc.set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic)

    if args.test:
        test_net(args, config)
    else:
        run_net(args, config, train_writer, val_writer)

if __name__ == '__main__':
    main()
