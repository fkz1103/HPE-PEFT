图数据包说明（export_figdata2.py 生成）
================================================================
[1] curves_all.csv / curve_<方法>.csv  —— Fig.7 收敛图
   列: method, epoch, train_loss, val_miou, val_acc, lr
   * 数据源: 训练日志的 [Validation] 行; val_miou 是 ShapeNetPart 验证集 Part-mIoU (0~1)
   * 采样密度天然不同(HPE-PEFT/Adapter/LoRA/VPT 的日志只记了 14~22 个 val 点;
     Shared/PointLoRA/Point-PEFT 每 epoch 都记, 150 点)。按原样折线画, 不要插值对齐。
   * legacy/curve_FullFT-LEGACY.csv 是旧评估协议(val mIoU 约 0.90), 与上面 7 条不同尺子, 禁止同图。
   * 已剔除日志里夹带的 epoch>150 伪点。

[2] eval8_val.csv / eval8_test.csv  —— Fig.8 权衡图
   列: method, params_M, ratio_pct, part_miou, inst_miou, cls_miou, acc, logged_best_val, delta_part_minus_logged
   * part_miou / acc 是百分数(81.32 表示 81.32%), params_M 单位百万, ratio_pct 是百分比。
   * 论文 Table VII 用 test 那份; val 那份仅供对照。
================================================================
