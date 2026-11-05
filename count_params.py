import sys
sys.path.insert(0, '/root/pointmae_peft/Point-MAE')
from utils import parser
from utils.config import cfg_from_yaml_file
from tools import builder

cfgs = {
 'ModelNet40': ['peft_hpe_modelnet40','peft_lora_modelnet40','peft_linear_modelnet40',
                'peft_adapter_b16_modelnet40','peft_ssf_modelnet40','peft_prompt_modelnet40'],
 'ScanObjectNN': ['finetune_scan_hardest_test','peft_adapter_b16','peft_hpe','peft_lora',
                  'peft_linearprobing','peft_ssf','peft_prompt'],
}
args = parser.get_args()
for ds, names in cfgs.items():
    print('='*20, ds, '='*20)
    for n in names:
        config = cfg_from_yaml_file(f'cfgs/{n}.yaml')
        model = builder.model_builder(config.model)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        method = config.model.get('peft_method', 'full')
        print(f'{n:<34} {method:<8} trainable={trainable/1e3:8.1f}K  total={total/1e6:6.2f}M  ratio={trainable/total*100:5.2f}%')
