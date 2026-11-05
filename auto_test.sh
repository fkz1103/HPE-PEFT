#!/bin/bash
set -o pipefail
cd /root/pointmae_peft/Point-MAE
export PYTHONPATH=/root/pointmae_peft/Point-MAE
LOG=/root/pointmae_peft/Point-MAE/auto_test.log

echo "========== [TEST] start $(date) ==========" >> $LOG

# T2 test: 1 epoch
python main.py --config cfgs/tmp_pointlora_scan.yaml --finetune_model \
    --ckpts ./pretrained/pretrain.pth --exp_name pointlora_scan_test --val_freq 1 \
    2>&1 | tee -a $LOG
RC=${PIPESTATUS[0]}
echo "[TEST] T2(1ep) exited rc=$RC $(date)" >> $LOG
if [ $RC -ne 0 ]; then echo "[TEST] ABORT at T2"; exit $RC; fi

# bank build: real full bank, reused by official run later
python build_pointpeft_bank.py \
    --config cfgs/pointpeft_modelnet40.yaml \
    --ckpts ./pretrained/pretrain.pth \
    --out ./pretrained/bank_modelnet40_train.pt \
    --batch_size 32 \
    --exp_name build_bank_modelnet40 \
    2>&1 | tee -a $LOG
RC=${PIPESTATUS[0]}
echo "[TEST] bank exited rc=$RC $(date)" >> $LOG
if [ $RC -ne 0 ]; then echo "[TEST] ABORT at bank"; exit $RC; fi

# T3 test: 1 epoch
python main.py --config cfgs/tmp_pointpeft_modelnet40.yaml --finetune_model \
    --ckpts ./pretrained/pretrain.pth --exp_name pointpeft_modelnet40_test --val_freq 1 \
    2>&1 | tee -a $LOG
RC=${PIPESTATUS[0]}
echo "[TEST] T3(1ep) exited rc=$RC $(date)" >> $LOG

echo "========== [TEST] done rc=$RC $(date) ==========" >> $LOG
exit $RC
