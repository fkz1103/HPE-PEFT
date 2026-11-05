import os, glob, re, datetime

ROOT = '/root/pointmae_peft/Point-MAE'
sets = {
  'ModelNet40': [
    ('HPE',     'experiments/peft_hpe_modelnet40/cfgs/peft_hpe_modelnet40'),
    ('LoRA',    'experiments/peft_lora_modelnet40/cfgs/peft_lora_modelnet40'),
    ('Linear',  'experiments/peft_linear_modelnet40/cfgs/peft_linear_modelnet40'),
    ('Adapter', 'experiments/peft_adapter_b16_modelnet40/cfgs/peft_adapter_b16_modelnet40'),
    ('SSF',     'experiments/peft_ssf_modelnet40/cfgs/peft_ssf_modelnet40'),
    ('VPT',     'experiments/peft_prompt_modelnet40/cfgs/peft_prompt_modelnet40'),
  ],
  'ScanObjectNN': [
    ('FullFT',  'experiments/finetune_scan_hardest_150ep/fair/fair_fullft_scan'),
    ('Adapter', 'experiments/peft_adapter_b16/cfgs/peft_adapter_b16'),
    ('HPE',     'experiments/peft_hpe/cfgs/peft_hpe'),
    ('LoRA',    'experiments/peft_lora/cfgs/peft_lora'),
    ('Linear',  'experiments/peft_linearprobing_fixed/cfgs/peft_linearprobing_fixed'),
    ('SSF',     'experiments/peft_ssf/cfgs/peft_ssf'),
    ('VPT',     'experiments/peft_prompt/cfgs/peft_prompt'),
  ],
}

for dset, lst in sets.items():
    print('=' * 26, dset, '=' * 26)
    print(f'{"Method":<10}{"Acc":>8}{"mAcc":>8}{"Params":>12}{"PeakMem":>10}{"TrainH":>8}')
    for name, d in lst:
        d = os.path.join(ROOT, d)
        acc = macc = float('nan')
        pf = os.path.join(d, 'per_class_acc.txt')
        if os.path.exists(pf):
            m = re.search(r'Overall\s+([\d.]+)\s+mAcc\s+([\d.]+)', open(pf).read())
            if m: acc, macc = float(m.group(1)), float(m.group(2))
        params, mem, train_s = '', 0.0, 0.0
        for f in glob.glob(os.path.join(d, '*.log')):
            txt = open(f, errors='ignore').read()
            if not params:
                mm = re.search(r'Trainable\s*=\s*([\d.]+[KM]?\s*params)', txt)
                if mm: params = mm.group(1)
            vals = [float(x) for x in re.findall(r'PeakMemory = ([\d.]+) GB', txt)]
            if vals: mem = max(mem, max(vals))
            train_s += sum(float(x) for x in re.findall(r'TrainTime = ([\d.]+) s', txt))
        print(f'{name:<10}{acc:>8.2f}{macc:>8.2f}{params:>12}{mem:>9.3f}G{train_s/3600:>8.2f}')
