import torch
import torch.nn as nn
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
import numpy as np
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils
from torchvision import transforms


train_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

test_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)


class SegMetric:
    def __init__(self, miou=0., acc=0.):
        self.miou = miou
        self.acc = acc

    def better_than(self, other):
        return self.miou > other.miou

    def state_dict(self):
        return {'miou': self.miou, 'acc': self.acc}


def compute_part_miou(pred_labels, gt_labels, num_parts=50):
    ious = []
    for p in range(num_parts):
        pred_mask = (pred_labels == p)
        gt_mask = (gt_labels == p)
        intersection = (pred_mask & gt_mask).sum().float()
        union = (pred_mask | gt_mask).sum().float()
        if union > 0:
            ious.append(intersection / union)
    return torch.tensor(ious).mean().item() if ious else 0.0


def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)

    # Build datasets
    train_sampler, train_dataloader = builder.dataset_builder(args, config.dataset.train)

    # Build validation dataset
    if hasattr(config.dataset, 'val'):
        val_sampler, val_dataloader = builder.dataset_builder(args, config.dataset.val)
    else:
        val_dataloader = None

    # Build model
    base_model = builder.model_builder(config.model)

    start_epoch = 0
    best_metrics = SegMetric(0., 0.)

    # Load pretrained checkpoint
    if args.ckpts is not None:
        base_model.load_model_from_ckpt(args.ckpts)
    else:
        print_log('Training from scratch!', logger=logger)

    if args.use_gpu:
        base_model.to(args.local_rank)

    if args.distributed:
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
        base_model = nn.parallel.DistributedDataParallel(
            base_model, device_ids=[args.local_rank % torch.cuda.device_count()])
    else:
        base_model = nn.DataParallel(base_model).cuda()

    # Optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    # Training loop
    base_model.zero_grad()
    metrics = SegMetric(0., 0.)
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        losses = AverageMeter(['loss', 'acc'])
        num_iter = 0
        n_batches = len(train_dataloader)
        npoints = config.npoints

        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1

            points = data[0].cuda()
            seg_label = data[1].cuda()

            # FPS sampling
            point_all = min(2400, points.size(1))
            fps_idx = pointnet2_utils.furthest_point_sample(points, point_all)
            fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
            points = pointnet2_utils.gather_operation(
                points.transpose(1, 2).contiguous(), fps_idx
            ).transpose(1, 2).contiguous()
            seg_label = torch.gather(seg_label, 1, fps_idx.long())

            points = train_transforms(points)

            logits = base_model(points)
            loss, acc = base_model.module.get_loss_acc(logits, seg_label)

            _loss = loss
            _loss.backward()

            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(
                        base_model.parameters(), config.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                acc = dist_utils.reduce_tensor(acc, args)

            losses.update([loss.item(), acc.item()])

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), epoch * n_batches + idx)
                train_writer.add_scalar('Loss/Batch/Acc', acc.item(), epoch * n_batches + idx)

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)

        print_log('[Training] EPOCH: %d Losses = %s lr = %.6f' %
                  (epoch, ['%.4f' % l for l in losses.avg()], optimizer.param_groups[0]['lr']), logger=logger)

        if epoch % args.val_freq == 0 and epoch != 0 and val_dataloader is not None:
            metrics = validate(base_model, val_dataloader, epoch, val_writer, args, config, logger=logger)
            if metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics,
                                        'ckpt-best', args, logger=logger)
                print_log("New best model! mIoU = %.4f, Acc = %.4f" % (metrics.miou, metrics.acc), logger=logger)

        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics,
                                'ckpt-last', args, logger=logger)

    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()


def validate(base_model, val_dataloader, epoch, val_writer, args, config, logger=None):
    base_model.eval()

    all_pred = []
    all_gt = []
    npoints = config.npoints

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(val_dataloader):
            points = data[0].cuda()
            seg_label = data[1].cuda()

            fps_idx = pointnet2_utils.furthest_point_sample(points, npoints).int()
            points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
            seg_label = torch.gather(seg_label, 1, fps_idx.long())

            logits = base_model(points)
            pred = logits.argmax(dim=1)

            all_pred.append(pred.detach().cpu())
            all_gt.append(seg_label.detach().cpu())

    all_pred = torch.cat(all_pred, dim=0)
    all_gt = torch.cat(all_gt, dim=0)

    acc = (all_pred == all_gt).sum().float() / all_gt.numel() * 100.
    miou = compute_part_miou(all_pred.flatten(), all_gt.flatten(), num_parts=50)

    print_log('[Validation] EPOCH: %d  mIoU = %.4f  Acc = %.4f' % (epoch, miou, acc), logger=logger)

    if val_writer is not None:
        val_writer.add_scalar('Metric/mIoU', miou, epoch)
        val_writer.add_scalar('Metric/Acc', acc, epoch)

    return SegMetric(miou, acc.item())


def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ...', logger=logger)

    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.ckpts, logger=logger)

    if args.use_gpu:
        base_model.to(args.local_rank)

    test(base_model, test_dataloader, args, config, logger=logger)


def test(base_model, test_dataloader, args, config, logger=None):
    base_model.eval()

    all_pred = []
    all_gt = []
    npoints = config.npoints

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            seg_label = data[1].cuda()

            fps_idx = pointnet2_utils.furthest_point_sample(points, npoints).int()
            points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
            seg_label = torch.gather(seg_label, 1, fps_idx.long())

            logits = base_model(points)
            pred = logits.argmax(dim=1)

            all_pred.append(pred.detach().cpu())
            all_gt.append(seg_label.detach().cpu())

    all_pred = torch.cat(all_pred, dim=0)
    all_gt = torch.cat(all_gt, dim=0)

    acc = (all_pred == all_gt).sum().float() / all_gt.numel() * 100.
    miou = compute_part_miou(all_pred.flatten(), all_gt.flatten(), num_parts=50)

    print_log('[TEST] mIoU = %.4f, Acc = %.4f' % (miou, acc), logger=logger)
