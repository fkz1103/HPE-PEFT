import os, re, glob, datetime
R = '/root/pointmae_peft/Point-MAE/experiments'
def ts(p): return datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime('%m-%d %H:%M')
def fld(t, n):
    m = re.search(r'(?m)^\s*' + n + r'\s*:\s*([^\n,}]+)', t)
    return m.group(1).strip().strip('\'"') if m else '-'

dirs = set()
for pat in ('/**/*.log', '/**/ckpt-best*.pth', '/**/per_class_acc.txt'):
    for p in glob.glob(R + pat, recursive=True):
        dirs.add(os.path.dirname(p))

warns = []
for d in sorted(dirs):
    rel = d.split('/experiments/')[-1]
    cfgp = os.path.join(d, 'config.yaml')
    print('=' * 104); print(rel)
    se = '-'
    if os.path.exists(cfgp):
        t = open(cfgp, errors='ignore').read()
        se = fld(t, 'epochs')
        print('   cfg@%s: sched.epochs=%s  max_epoch=%s  lr=%s  freeze_backbone=%s  peft_method=%s'
              % (ts(cfgp), se, fld(t, 'max_epoch'), fld(t, 'lr'), fld(t, 'freeze_backbone'), fld(t, 'peft_method')))
    else:
        print('   cfg: *** 无 config.yaml ***')
    ck = sorted(glob.glob(d + '/ckpt-best*.pth'))
    print('   ckpt: ' + (', '.join('%s@%s %.1fMB' % (os.path.basename(c), ts(c), os.path.getsize(c) / 1e6) for c in ck) or '-'))
    ev = os.path.join(d, 'per_class_acc.txt')
    print('   eval: ' + ((open(ev).readline().strip() + '  [' + ts(ev) + ']') if os.path.exists(ev) else '*** 未评估 ***'))
    for f in sorted(glob.glob(d + '/*.log')):
        txt = open(f, errors='ignore').read()
        vs = [(int(a), float(b)) for a, b in re.findall(r'EPOCH:\s*(\d+)\D+acc\s*=\s*([\d.]+)', txt)]
        m = re.search(r'Trainable\s*=\s*([\d.]+[KM]?\s*params)', txt)
        tr = m.group(1) if m else '-'
        if not vs:
            print('   log %s@%s: 无验证记录(评估/中断脚本)  Trainable=%s' % (os.path.basename(f), ts(f), tr)); continue
        seen = max(e for e, _ in vs)
        b = max(vs, key=lambda x: x[1])
        l3 = [v for _, v in vs[-3:]]
        gap = b[1] - sum(l3) / len(l3)
        fl = ''
        if se != '-' and str(seen) != se: fl += '  <<< WARN 调度%s!=实跑%s' % (se, seen)
        if gap > 1.0:                     fl += '  <<< WARN 尾段回落%.2f(未退火指纹)' % gap
        if b[0] < 0.5 * seen:             fl += '  <<< WARN 早期封顶(best@ep%d/%d)' % (b[0], seen)
        print('   log %s@%s: seen=%d best=%.4f@ep%d last3=%s gap=%.2f Trainable=%s%s'
              % (os.path.basename(f), ts(f), seen, b[1], b[0], ['%.2f' % x for x in l3], gap, tr, fl))
        if fl: warns.append((rel, os.path.basename(f), fl.strip()))

print('\n' + '=' * 104); print('WARN 汇总：')
for rel, fn, fl in warns:
    print('  %-58s %-26s %s' % (rel, fn, fl))
if not warns: print('  （无）')
