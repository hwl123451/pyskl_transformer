# import torch
#
#
# def merge_weights(rgb_path, pose_path, save_path):
#     print(f"正在合并权重...")
#     print(f"RGB 权重: {rgb_path}")
#     print(f"Pose 权重: {pose_path}")
#
#     # 1. 加载两个权重文件
#     rgb_state = torch.load(rgb_path, map_location='cpu')
#     pose_state = torch.load(pose_path, map_location='cpu')
#
#     # 提取 state_dict
#     rgb_dict = rgb_state['state_dict']
#     pose_dict = pose_state['state_dict']
#
#     new_dict = {}
#
#     # 2. 处理 RGB 分支的权重 (添加 'backbone.rgb_pathway.' 前缀)
#     for k, v in rgb_dict.items():
#         if k.startswith('backbone.'):
#             # 例如: backbone.layer1 -> backbone.rgb_pathway.layer1
#             new_k = k.replace('backbone.', 'backbone.rgb_pathway.')
#             new_dict[new_k] = v
#         elif k.startswith('cls_head.'):
#             # 例如: cls_head.fc -> cls_head.fc_rgb
#             new_k = k.replace('cls_head.', 'cls_head.fc_rgb.') if 'fc' in k else k
#             new_dict[new_k] = v
#
#     # 3. 处理 Pose 分支的权重 (添加 'backbone.pose_pathway.' 前缀)
#     for k, v in pose_dict.items():
#         if k.startswith('backbone.'):
#             # 例如: backbone.layer1 -> backbone.pose_pathway.layer1
#             new_k = k.replace('backbone.', 'backbone.pose_pathway.')
#             new_dict[new_k] = v
#         elif k.startswith('cls_head.'):
#             # 例如: cls_head.fc -> cls_head.fc_pose
#             new_k = k.replace('cls_head.', 'cls_head.fc_pose.') if 'fc' in k else k
#             new_dict[new_k] = v
#
#     # 4. 保存合并后的权重
#     torch.save(dict(state_dict=new_dict), save_path)
#     print(f"✅ 合并完成！权重已保存至: {save_path}")
#
#
# if __name__ == '__main__':
#     # ================= 修改这里 =================
#     # 指向你训练好的最佳权重路径 (通常在 work_dirs 下)
#     rgb_checkpoint = '/media/zjq/6CDE775EDE771F8E/lxk/video_rgbPose/work_dirs/rgbpose_conv3d/rgb_only_hwlY/best_top1_acc_epoch_17.pth'
#     pose_checkpoint = '/media/zjq/6CDE775EDE771F8E/lxk/video_rgbPose/work_dirs/rgbpose_conv3d/pose_only_hwlY/best_top1_acc_epoch_40.pth'  # 请确认你的 pose 权重路径
#
#     output_checkpoint = 'rgbpose_conv3d_init_hwlY.pth'
#     # ===========================================
#
#     merge_weights(rgb_checkpoint, pose_checkpoint, output_checkpoint)


#twice
import torch
import copy as cp
from collections import OrderedDict


def padding(weight, new_shape):
    """
    官方提供的 Padding 函数：用于处理因侧向连接导致的通道数增加。
    将原权重复制过去，新增的通道部分填 0。
    """
    new_weight = weight.new_zeros(new_shape)
    # 也就是只复制前 N 个通道 (原通道数)
    new_weight[:, :weight.shape[1]] = weight
    return new_weight


def merge_weights(rgb_path, pose_path, save_path):
    print(f"🔄 正在合并权重...")
    print(f"   RGB 源: {rgb_path}")
    print(f"   Pose 源: {pose_path}")

    # 1. 加载权重
    rgb_state = torch.load(rgb_path, map_location='cpu')
    pose_state = torch.load(pose_path, map_location='cpu')

    rgb_ckpt = rgb_state['state_dict']
    pose_ckpt = pose_state['state_dict']

    # 2. 官方重命名逻辑 (注意是 rgb_path 而不是 rgb_pathway)
    # 同时处理 backbone 和 head (fc_cls -> fc_rgb)
    new_rgb_ckpt = {}
    for k, v in rgb_ckpt.items():
        new_k = k.replace('backbone', 'backbone.rgb_path').replace('fc_cls', 'fc_rgb')
        # 你的脚本之前处理 cls_head 的逻辑可能有点复杂，官方这句 replace 简单有效
        if k.startswith('cls_head'):
            # 确保 cls_head.fc_cls 变成了 cls_head.fc_rgb
            new_k = k.replace('fc_cls', 'fc_rgb')
        new_rgb_ckpt[new_k] = v

    new_pose_ckpt = {}
    for k, v in pose_ckpt.items():
        new_k = k.replace('backbone', 'backbone.pose_path').replace('fc_cls', 'fc_pose')
        if k.startswith('cls_head'):
            new_k = k.replace('fc_cls', 'fc_pose')
        new_pose_ckpt[new_k] = v

    # 3. 合并字典
    ckpt = {}
    ckpt.update(new_rgb_ckpt)
    ckpt.update(new_pose_ckpt)

    # =========================================================
    # 4. [关键步骤] Padding: 处理侧向连接导致的维度不匹配
    # =========================================================
    print("⚡ 正在执行 Padding (适配侧向连接)...")

    # RGB 分支的 Padding (Layer 3 和 Layer 4 接收侧向连接)
    # 注意：这里的数字 (256, 640) 等是基于 ResNet50 SlowOnly 的标准结构

    name = 'backbone.rgb_path.layer3.0.conv1.conv.weight'
    if name in ckpt:
        ckpt[name] = padding(ckpt[name], (256, 640, 3, 1, 1))

    name = 'backbone.rgb_path.layer3.0.downsample.conv.weight'
    if name in ckpt:
        ckpt[name] = padding(ckpt[name], (1024, 640, 1, 1, 1))

    name = 'backbone.rgb_path.layer4.0.conv1.conv.weight'
    if name in ckpt:
        ckpt[name] = padding(ckpt[name], (512, 1280, 3, 1, 1))

    name = 'backbone.rgb_path.layer4.0.downsample.conv.weight'
    if name in ckpt:
        ckpt[name] = padding(ckpt[name], (2048, 1280, 1, 1, 1))

    # Pose 分支的 Padding (Layer 2 和 Layer 3 接收侧向连接)
    name = 'backbone.pose_path.layer2.0.conv1.conv.weight'
    if name in ckpt:
        ckpt[name] = padding(ckpt[name], (64, 160, 3, 1, 1))

    name = 'backbone.pose_path.layer2.0.downsample.conv.weight'
    if name in ckpt:
        ckpt[name] = padding(ckpt[name], (256, 160, 1, 1, 1))

    name = 'backbone.pose_path.layer3.0.conv1.conv.weight'
    if name in ckpt:
        ckpt[name] = padding(ckpt[name], (128, 320, 3, 1, 1))

    name = 'backbone.pose_path.layer3.0.downsample.conv.weight'
    if name in ckpt:
        ckpt[name] = padding(ckpt[name], (512, 320, 1, 1, 1))

    # 5. 保存
    final_dict = OrderedDict(ckpt)
    torch.save(dict(state_dict=final_dict), save_path)
    print(f"✅ 合并成功！文件已保存至: {save_path}")
    print(f"   (包含了 {len(final_dict)} 个权重键值对)")


if __name__ == '__main__':
    # 务必确认这里的路径是正确的（指向那个 90% 的 Pose 和 80% 的 RGB）
    rgb_checkpoint = '/media/zjq/6CDE775EDE771F8E/lxk/video_rgbPose/work_dirs/rgbpose_conv3d/rgb_only_hwlY/best_top1_acc_epoch_17.pth'

    # 指向 pose_only 目录 (90.7% 的那个)
    pose_checkpoint = '/media/zjq/6CDE775EDE771F8E/lxk/video_rgbPose/work_dirs/rgbpose_conv3d/pose_only_hwlY/best_top1_acc_epoch_40.pth'

    output_checkpoint = 'rgbpose_conv3d_init_FINAL.pth'

    merge_weights(rgb_checkpoint, pose_checkpoint, output_checkpoint)