import os
import sys
import numpy as np
import torch

# 删除 hpenext_kd 路径，避免导入错误的 openpoints.datasets
sys.path = [
    p for p in sys.path
    if "hpenext" not in p
]

# 确保当前 Point-MAE 项目路径位于 sys.path 首位
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)

sys.path.insert(0, PROJECT_ROOT)

from tools import builder
from utils import parser
from utils.config import get_config
from utils.logger import get_root_logger
from pointnet2_ops import pointnet2_utils


def main():
    args = parser.get_args()
    args.test = True
    args.use_gpu = torch.cuda.is_available()
    args.distributed = False

    logger = get_root_logger(
        log_file=None,
        name="shapenetpart_feature_export"
    )

    config = get_config(args, logger=logger)

    # 测试集批次大小
    if not hasattr(config.dataset.test.others, "bs"):
        config.dataset.test.others.bs = 16

    print("[1/5] Building ShapeNetPart test dataloader...")
    _, test_dataloader = builder.dataset_builder(
        args,
        config.dataset.test
    )

    print("[2/5] Building model...")
    base_model = builder.model_builder(config.model)

    print("[3/5] Loading checkpoint...")
    builder.load_model(
        base_model,
        args.ckpts,
        logger=logger
    )

    if args.use_gpu:
        base_model = base_model.cuda()

    base_model.eval()

    # 捕获最终分类卷积层的输入：
    # seg_head[-1] 是 Conv1d(256, 50, 1)，
    # 它的输入即为 256 维点级特征。
    feature_buffer = {}

    def capture_features(module, module_input, module_output):
        feature_buffer["point_features"] = module_input[0].detach()

    target_head = base_model.seg_head[-1]
    hook_handle = target_head.register_forward_hook(capture_features)

    # 保存物体级特征
    all_shape_features = []
    all_shape_categories = []
    all_shape_ids = []

    # 保存点级特征
    all_point_features = []
    all_part_labels = []
    all_category_labels = []
    all_shape_ids_point = []
    all_point_indices = []
    all_predicted_labels = []
    all_point_coordinates = []

    # taxonomy ID 到整数类别 ID 的映射
    category_to_index = {}

    # 每个点云固定采样的点数
    points_per_shape = 32

    print("[4/5] Extracting features...")

    with torch.no_grad():
        for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(
            test_dataloader
        ):
            points = data[0].cuda(non_blocking=True)
            part_labels = data[1].cuda(non_blocking=True)

            # 与原 runner_seg.py 保持一致：
            # 使用 FPS 得到 2048 个点，并同步更新部件标签。
            npoints = config.npoints
            fps_idx = pointnet2_utils.furthest_point_sample(
                points,
                npoints
            ).int()

            points = pointnet2_utils.gather_operation(
                points.transpose(1, 2).contiguous(),
                fps_idx
            ).transpose(1, 2).contiguous()

            part_labels = torch.gather(
                part_labels,
                1,
                fps_idx.long()
            )

            logits = base_model(points)
            point_features = feature_buffer["point_features"]
            predictions = logits.argmax(dim=1)

            # point_features: [B, 256, N]
            # 转换为 [B, N, 256]
            point_features = point_features.transpose(1, 2).contiguous()

            batch_size, num_points, feature_dim = point_features.shape

            # 物体级特征：
            # 对点级特征进行平均池化和最大池化，再进行拼接
            shape_mean = point_features.mean(dim=1)
            shape_max = point_features.max(dim=1).values
            shape_features = torch.cat(
                [shape_mean, shape_max],
                dim=1
            )

            all_shape_features.append(
                shape_features.cpu().numpy()
            )

            # 统一处理 taxonomy ID
            taxonomy_ids = [
                str(x) for x in taxonomy_ids
            ]

            category_indices = []
            for category_name in taxonomy_ids:
                if category_name not in category_to_index:
                    category_to_index[category_name] = len(
                        category_to_index
                    )
                category_indices.append(
                    category_to_index[category_name]
                )

            all_shape_categories.append(
                np.asarray(category_indices, dtype=np.int64)
            )

            all_shape_ids.append(
                np.asarray(
                    [str(x) for x in model_ids],
                    dtype=str
                )
            )

            # 点级特征固定采样 32 个点
            if num_points <= points_per_shape:
                sample_indices = torch.arange(
                    num_points,
                    device=points.device
                )
            else:
                sample_indices = torch.linspace(
                    0,
                    num_points - 1,
                    points_per_shape,
                    device=points.device
                ).long()

            sampled_features = point_features[:, sample_indices, :]
            sampled_labels = part_labels[:, sample_indices]
            sampled_predictions = predictions[:, sample_indices]
            sampled_points = points[:, sample_indices, :]

            all_point_features.append(
                sampled_features.cpu().numpy()
            )

            all_part_labels.append(
                sampled_labels.cpu().numpy()
            )

            all_predicted_labels.append(
                sampled_predictions.cpu().numpy()
            )

            all_point_coordinates.append(
                sampled_points.cpu().numpy()
            )

            all_point_indices.append(
                sample_indices.unsqueeze(0)
                .expand(batch_size, -1)
                .cpu()
                .numpy()
            )

            # 将每个点映射到其所属物体类别
            category_indices_tensor = torch.tensor(
                category_indices,
                device=points.device,
                dtype=torch.long
            )

            point_category_labels = category_indices_tensor.unsqueeze(
                1
            ).expand(
                -1,
                sample_indices.numel()
            )

            all_category_labels.append(
                point_category_labels.cpu().numpy()
            )

            # 为每个点保存所属点云的编号
            shape_ids_batch = np.arange(
                batch_idx * batch_size,
                (batch_idx + 1) * batch_size,
                dtype=np.int64
            )

            point_shape_ids = np.repeat(
                shape_ids_batch,
                sample_indices.numel()
            )

            all_shape_ids_point.append(point_shape_ids)

            if (batch_idx + 1) % 20 == 0:
                print(
                    f"Processed {batch_idx + 1} batches..."
                )

    hook_handle.remove()

    print("[5/5] Saving extracted features...")

    output_dir = os.path.join(
        PROJECT_ROOT,
        "visualization",
        "shapenetpart_hpe"
    )
    os.makedirs(output_dir, exist_ok=True)

    # 合并物体级数据
    shape_features = np.concatenate(
        all_shape_features,
        axis=0
    )

    shape_categories = np.concatenate(
        all_shape_categories,
        axis=0
    )

    shape_ids = np.concatenate(
        all_shape_ids,
        axis=0
    )

    # 合并点级数据
    point_features = np.concatenate(
        all_point_features,
        axis=0
    )

    part_labels = np.concatenate(
        all_part_labels,
        axis=0
    ).reshape(-1)

    predicted_labels = np.concatenate(
        all_predicted_labels,
        axis=0
    ).reshape(-1)

    point_coordinates = np.concatenate(
        all_point_coordinates,
        axis=0
    ).reshape(-1, 3)

    point_indices = np.concatenate(
        all_point_indices,
        axis=0
    ).reshape(-1)

    category_labels = np.concatenate(
        all_category_labels,
        axis=0
    ).reshape(-1)

    point_shape_ids = np.concatenate(
        all_shape_ids_point,
        axis=0
    )

    category_names = np.asarray(
        [
            name for name, index in sorted(
                category_to_index.items(),
                key=lambda item: item[1]
            )
        ],
        dtype=str
    )

    output_path = os.path.join(
        output_dir,
        "shapenetpart_hpe_features.npz"
    )

    np.savez_compressed(
        output_path,
        shape_features=shape_features,
        shape_categories=shape_categories,
        shape_ids=shape_ids,
        point_features=point_features,
        part_labels=part_labels,
        predicted_labels=predicted_labels,
        category_labels=category_labels,
        point_shape_ids=point_shape_ids,
        point_indices=point_indices,
        point_coordinates=point_coordinates,
        category_names=category_names
    )

    print("\nFeature export completed.")
    print("Output:", output_path)
    print("shape_features:", shape_features.shape)
    print("point_features:", point_features.shape)
    print("part_labels:", part_labels.shape)
    print("category_labels:", category_labels.shape)
    print("category_names:", category_names)


if __name__ == "__main__":
    main()
