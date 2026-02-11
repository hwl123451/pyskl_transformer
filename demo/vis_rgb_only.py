import torch
import cv2
import numpy as np
import mmcv
from pyskl.apis import init_recognizer
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ================= 1. 请替换这里的路径 =================
config_file = 'E:\\aisi\\video_rgbPose\\configs\\rgbpose_conv3d\\rgbpose_conv3d.py'  # 例如: configs/rgbpose_conv3d/....py
checkpoint_file = 'E:\\aisi\\video_rgbPose\\work_dirs\\rgbpose_conv3d\\rgbpose_conv3d\\best_rgb_top1_acc_epoch_10.pth'  # 训练好的模型权重
video_path = 'E:\\openmmlab\\mmaction2\\demo\\S001C001P001R001A010_rgb.avi'  # 你的 NTU 视频文件路径


# ======================================================

def reshape_transform(tensor):
    """
    辅助函数：让 Grad-CAM 理解特征图的维度。
    """
    # 情况 A: 如果特征图已经是 4D [N, C, H, W] (例如 [8, 2048, 7, 7])
    # 这说明模型内部已经把 B 和 T 合并了，直接返回即可，Grad-CAM 能看懂。
    if tensor.ndim == 4:
        return tensor

    # 情况 B: 如果特征图是 5D [B, C, T, H, W] (传统的 3D CNN)
    # 我们才需要把它手动展平
    if tensor.ndim == 5:
        B, C, T, H, W = tensor.shape
        return tensor.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)


class RGBOnlyWrapper(torch.nn.Module):
    def __init__(self, recognizer):
        super().__init__()
        self.recognizer = recognizer

    def forward(self, x):
        # x 进来时的形状: [B*T, C, H, W] -> [8, 3, 224, 224]
        # 我们需要把它还原成: [B, C, T, H, W] -> [1, 3, 8, 224, 224]

        # 1. 维度还原
        if x.ndim == 4:
            BT, C, H, W = x.shape
            T = 8  # 你的 clip_len
            B = BT // T

            # 关键修正：
            # 1. 先变回 [B, T, C, H, W] (因为我们在 main 里是先把 T 换到前面的)
            # 2. 再 permute 回 [B, C, T, H, W] (模型需要的格式)
            x_5d = x.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        else:
            x_5d = x

        # 2. 构造 Pose (5D)
        N, C, T, H, W = x_5d.shape
        dummy_pose = torch.zeros((N, 17, T, H, W)).to(x.device)

        # 3. 提取特征
        feat = self.recognizer.backbone(imgs=x_5d, heatmap_imgs=dummy_pose)

        # 4. 计算分数
        cls_score = self.recognizer.cls_head(feat)

        if isinstance(cls_score, dict):
            score = cls_score['rgb'] if 'rgb' in cls_score else sum(cls_score.values())
        else:
            score = cls_score

        # 5. 输出展平
        if x.ndim == 4:
            return score.expand(BT, -1)

        return score


def preprocess_video(video_path, clip_len=8):
    """简单的视频预处理，对齐训练时的 Transform"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()

    # 均匀采样 clip_len 帧
    indices = np.linspace(0, len(frames) - 1, clip_len).astype(int)
    sampled_frames = [frames[i] for i in indices]

    # Resize 和 Normalize (参考你的 Config)
    # Resize: (240, 320) -> CenterCrop -> Norm
    processed_frames = []
    viz_frames = []  # 用于最后画图的原始图

    for frame in sampled_frames:
        # Resize 到 224x224 (假设训练是这个尺寸)
        frame = cv2.resize(frame, (224, 224))
        viz_frames.append(frame)  # 保存 BGR 用于画图

        # 转 RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Normalize
        frame_norm = (frame_rgb - np.array([123.675, 116.28, 103.53])) / np.array([58.395, 57.12, 57.375])
        processed_frames.append(frame_norm)

    # [T, H, W, C] -> [C, T, H, W]
    input_tensor = torch.tensor(np.array(processed_frames), dtype=torch.float32)
    input_tensor = input_tensor.permute(3, 0, 1, 2).unsqueeze(0)  # 增加 Batch 维

    return input_tensor, viz_frames


def main():
    device = 'cuda:0'
    model = init_recognizer(config_file, checkpoint_file, device=device)
    model.eval()

    wrapper_model = RGBOnlyWrapper(model)
    target_layers = [model.backbone.rgb_path.feature_extract.model.layer4[-1]]

    input_tensor, viz_frames = preprocess_video(video_path, clip_len=8)
    input_tensor = input_tensor.to(device) # Shape: [1, 3, 8, 224, 224]

    # 初始化 GradCAM
    cam = GradCAM(model=wrapper_model,
                  target_layers=target_layers,
                  reshape_transform=reshape_transform)

    # ==================== 关键修改开始 ====================
    # 目的：把 [1, 3, 8, 224, 224] 变成 [8, 3, 224, 224]
    # 步骤：
    # 1. permute: 把时间 T(8) 换到前面 -> [1, 8, 3, 224, 224]
    # 2. reshape: 合并 B 和 T -> [8, 3, 224, 224]
    # 这样每一张图才是完整的 RGB 帧，不会乱码
    input_2d = input_tensor.permute(0, 2, 1, 3, 4).reshape(-1, 3, 224, 224)
    # ==================== 关键修改结束 ====================

    grayscale_cam = cam(input_tensor=input_2d, targets=None)

    # 生成图
    concat_img = []
    for i in range(len(viz_frames)):
        rgb_img = cv2.cvtColor(viz_frames[i], cv2.COLOR_BGR2RGB) / 255.0
        heatmap = grayscale_cam[i, :, :]
        visualization = show_cam_on_image(rgb_img, heatmap, use_rgb=True)
        concat_img.append(visualization)

    final_result = np.hstack(concat_img)
    cv2.imwrite('rgb_attention_strip.jpg', cv2.cvtColor(final_result, cv2.COLOR_RGB2BGR))
    print(f"成功！结果已保存为 rgb_attention_strip.jpg")


if __name__ == '__main__':
    main()