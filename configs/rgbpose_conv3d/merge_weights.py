import torch


def merge_weights(rgb_path, pose_path, save_path):
    print(f"正在合并权重...")
    print(f"RGB 权重: {rgb_path}")
    print(f"Pose 权重: {pose_path}")

    # 1. 加载两个权重文件
    rgb_state = torch.load(rgb_path, map_location='cpu')
    pose_state = torch.load(pose_path, map_location='cpu')

    # 提取 state_dict
    rgb_dict = rgb_state['state_dict']
    pose_dict = pose_state['state_dict']

    new_dict = {}

    # 2. 处理 RGB 分支的权重 (添加 'backbone.rgb_pathway.' 前缀)
    for k, v in rgb_dict.items():
        if k.startswith('backbone.'):
            # 例如: backbone.layer1 -> backbone.rgb_pathway.layer1
            new_k = k.replace('backbone.', 'backbone.rgb_pathway.')
            new_dict[new_k] = v
        elif k.startswith('cls_head.'):
            # 例如: cls_head.fc -> cls_head.fc_rgb
            new_k = k.replace('cls_head.', 'cls_head.fc_rgb.') if 'fc' in k else k
            new_dict[new_k] = v

    # 3. 处理 Pose 分支的权重 (添加 'backbone.pose_pathway.' 前缀)
    for k, v in pose_dict.items():
        if k.startswith('backbone.'):
            # 例如: backbone.layer1 -> backbone.pose_pathway.layer1
            new_k = k.replace('backbone.', 'backbone.pose_pathway.')
            new_dict[new_k] = v
        elif k.startswith('cls_head.'):
            # 例如: cls_head.fc -> cls_head.fc_pose
            new_k = k.replace('cls_head.', 'cls_head.fc_pose.') if 'fc' in k else k
            new_dict[new_k] = v

    # 4. 保存合并后的权重
    torch.save(dict(state_dict=new_dict), save_path)
    print(f"✅ 合并完成！权重已保存至: {save_path}")


if __name__ == '__main__':
    # ================= 修改这里 =================
    # 指向你训练好的最佳权重路径 (通常在 work_dirs 下)
    rgb_checkpoint = '/media/zjq/6CDE775EDE771F8E/lxk/video_rgbPose/work_dirs/rgbpose_conv3d/rgb_only_hwlY/best_top1_acc_epoch_17.pth'
    pose_checkpoint = '/media/zjq/6CDE775EDE771F8E/lxk/video_rgbPose/work_dirs/rgbpose_conv3d/pose_only_hwlY/best_top1_acc_epoch_40.pth'  # 请确认你的 pose 权重路径

    output_checkpoint = 'rgbpose_conv3d_init_hwlY.pth'
    # ===========================================

    merge_weights(rgb_checkpoint, pose_checkpoint, output_checkpoint)