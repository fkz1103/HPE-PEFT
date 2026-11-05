#!/bin/bash
# ================================================================
# 两任务流水线:任务1 PointLoRA(150ep) → 校验 → 任务2 Point-PEFT(150ep)
# 由 nohup 启动,SSH 断线不影响;实时查看:tail -f chain.log
# 取消:pkill -f run_two_seg.sh
# ================================================================
cd /root/pointmae_peft/Point-MAE

echo "[$(date '+%F %T')] [pipeline] ===== 任务1/2: PointLoRA ShapeNetPart ====="
rm -rf experiments/shapenet_part_seg_pointlora
CUDA_VISIBLE_DEVICES=0 python main_seg.py \
  --config cfgs/shapenet_part_seg_pointlora.yaml \
  --exp_name shapenet_part_seg_pointlora \
  --ckpts pretrained/pretrain.pth \
  --num_workers 8

# --- 校验任务1跑满 150 epoch,没跑满(崩溃)绝不进入任务2 ---
if ! grep -rqE "EPOCH: ?149" experiments/shapenet_part_seg_pointlora 2>/dev/null; then
  echo "[$(date '+%F %T')] [pipeline] ✗ PointLoRA 未跑满 150 epoch(疑似崩溃),流水线停止!"
  exit 1
fi
echo "[$(date '+%F %T')] [pipeline] ✓ PointLoRA 150 epoch 完成"
sleep 30   # 等显存完全释放

# --- 任务2就位检查:yaml + 模型分支 + 10秒冒烟测试,每5分钟重查,最多6小时 ---
echo "[$(date '+%F %T')] [pipeline] ===== 任务2/2: Point-PEFT ShapeNetPart ====="
ready=0
for i in $(seq 1 72); do
  if [ -f cfgs/shapenet_part_seg_pointpeft.yaml ] \
     && [ -f test_pointpeft_seg.py ] \
     && python test_pointpeft_seg.py > /tmp/peft_smoke.log 2>&1; then
    ready=1; break
  fi
  echo "[$(date '+%F %T')] [pipeline] Point-PEFT 代码未就位(第 $i/72 次检查),5 分钟后重查..."
  sleep 300
done
if [ "$ready" -ne 1 ]; then
  echo "[$(date '+%F %T')] [pipeline] ✗ 等待超时。手动启动命令:"
  echo "  CUDA_VISIBLE_DEVICES=0 python main_seg.py --config cfgs/shapenet_part_seg_pointpeft.yaml --exp_name shapenet_part_seg_pointpeft --ckpts pretrained/pretrain.pth --num_workers 8"
  echo "[pipeline] 冒烟测试报错日志: cat /tmp/peft_smoke.log"
  exit 1
fi
echo "[$(date '+%F %T')] [pipeline] ✓ 冒烟测试通过,启动 Point-PEFT..."
rm -rf experiments/shapenet_part_seg_pointpeft
CUDA_VISIBLE_DEVICES=0 python main_seg.py \
  --config cfgs/shapenet_part_seg_pointpeft.yaml \
  --exp_name shapenet_part_seg_pointpeft \
  --ckpts pretrained/pretrain.pth \
  --num_workers 8
echo "[$(date '+%F %T')] [pipeline] ✓✓ 两个任务全部完成!"
