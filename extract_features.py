"""一次 GPU 推理，保存特征到 .npy 文件，后续画图用 CPU 即可"""
import torch, numpy as np, os, sys
sys.path.append('/root/pointmae_peft/Point-MAE/extensions/chamfer_dist')
sys.path = ['/root/pointmae_peft/Point-MAE'] + [p for p in sys.path if 'hpenext_kd' not in p]

from utils import parser
from utils.logger import get_logger
from utils.config import get_config
from tools import builder
from tqdm import tqdm
import logging

config_path, ckpt_path, exp_name, out_name = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

sys.argv = ['extract.py', '--config', config_path, '--exp_name', exp_name,
            '--test', '--ckpts', ckpt_path, '--launcher', 'none']
args = parser.get_args()
args.distributed = False
args.use_gpu = torch.cuda.is_available()
args.world_size = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = get_logger('extract_logger')
logger.setLevel(logging.ERROR)
config = get_config(args, logger=logger)

config.dataset.test.others.bs = 1
_, test_dataloader = builder.dataset_builder(args, config.dataset.test)

base_model = builder.model_builder(config.model)
builder.load_model(base_model, args.ckpts, logger=logger)
base_model.to(device)
base_model = torch.nn.DataParallel(base_model)
base_model.eval()

# Hook 提取特征
hook_features = []
def hook_fn(m, i, o):
    hook_features.append(i[0].detach().cpu())
handle = base_model.module.cls_head_finetune[0].register_forward_hook(hook_fn)

all_logits, all_labels, all_preds = [], [], []
with torch.no_grad():
    for _, _, data in tqdm(test_dataloader, desc='Extracting'):
        points = data[0].to(device)
        label = data[1].to(device)
        from utils import misc
        points = misc.fps(points, config.npoints)
        logits = base_model(points)
        all_logits.append(logits.cpu())
        all_labels.append(label.cpu())
        all_preds.append(logits.argmax(-1).cpu())

handle.remove()

os.makedirs(f'visualization/{exp_name}', exist_ok=True)
np.save(f'visualization/{exp_name}/{out_name}_features.npy', torch.cat(hook_features).numpy())
np.save(f'visualization/{exp_name}/{out_name}_logits.npy', torch.cat(all_logits).numpy())
np.save(f'visualization/{exp_name}/{out_name}_labels.npy', torch.cat(all_labels).numpy())
np.save(f'visualization/{exp_name}/{out_name}_preds.npy', torch.cat(all_preds).numpy())
print(f"Saved to visualization/{exp_name}/{out_name}_*.npy")
