#!/bin/bash
set -o pipefail
cd /root/pointmae_peft/Point-MAE
export PYTHONPATH=/root/pointmae_peft/Point-MAE
LOG=/root/pointmae_peft/Point-MAE/auto_night.log

echo "========== [AUTO] night run started $(date) ==========" >> $LOG

rm -rf experiments/pointlora_scan experiments/pointpeft_modelnet40

# --- T2: PointLoRA Scan (150 ep) ---
echo "[AUTO] T2 PointLoRA-Scan starting $(date)" >> $LOG
python main.py --config cfgs/pointlora_scan.yaml --finetune_model \
    --ckpts ./pretrained/pretrain.pth --exp_name pointlora_scan --val_freq 1 \
    2>&1 | tee -a $LOG
RC=${PIPESTATUS[0]}
echo "[AUTO] T2 exited rc=$RC $(date)" >> $LOG
if [ $RC -ne 0 ]; then
    echo "[AUTO] ABORT: T2 failed, T3 NOT started" >> $LOG
    exit $RC
fi

# --- T3a: build ModelNet40 memory bank ---
echo "[AUTO] T3a build bank starting $(date)" >> $LOG
python build_pointpeft_bank.py \
    --config cfgs/pointpeft_modelnet40.yaml \
    --ckpts ./pretrained/pretrain.pth \
    --out ./pretrained/bank_modelnet40_train.pt \
    --batch_size 32 \
    --exp_name build_bank_modelnet40 \
    2>&1 | tee -a $LOG
RC=${PIPESTATUS[0]}
echo "[AUTO] bank build exited rc=$RC $(date)" >> $LOG
if [ $RC -ne 0 ]; then
    echo "[AUTO] ABORT: bank build failed, T3b NOT started" >> $LOG
    exit $RC
fi

# --- T3b: Point-PEFT ModelNet40 (150 ep) ---
echo "[AUTO] T3b PointPEFT-ModelNet40 starting $(date)" >> $LOG
python main.py --config cfgs/pointpeft_modelnet40.yaml --finetune_model \
    --ckpts ./pretrained/pretrain.pth --exp_name pointpeft_modelnet40 --val_freq 1 \
    2>&1 | tee -a $LOG
RC=${PIPESTATUS[0]}
echo "[AUTO] T3b exited rc=$RC $(date)" >> $LOG

echo "========== [AUTO] night run finished rc=$RC $(date) ==========" >> $LOG
exit $RC
