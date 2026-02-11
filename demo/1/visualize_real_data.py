import numpy as np
import matplotlib.pyplot as plt
import pickle


# ==========================================
# 1. 核心热图生成函数 (保持不变)
# ==========================================
def generate_gaussian_heatmap(img_h, img_w, centers, sigma, max_values):
    arr = np.zeros((img_h, img_w), dtype=np.float32)
    for center, max_value in zip(centers, max_values):
        if max_value < 1e-3: continue

        mu_x, mu_y = center[0], center[1]
        st_x = max(int(mu_x - 3 * sigma), 0)
        ed_x = min(int(mu_x + 3 * sigma) + 1, img_w)
        st_y = max(int(mu_y - 3 * sigma), 0)
        ed_y = min(int(mu_y + 3 * sigma) + 1, img_h)
        x = np.arange(st_x, ed_x, 1, np.float32)
        y = np.arange(st_y, ed_y, 1, np.float32)

        if not (len(x) and len(y)): continue
        y = y[:, None]
        patch = np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / 2 / sigma ** 2)
        patch = patch * max_value
        arr[st_y:ed_y, st_x:ed_x] = np.maximum(arr[st_y:ed_y, st_x:ed_x], patch)
    return arr


# ==========================================
# 2. 加载真实数据的函数
# ==========================================
def load_real_sample(pkl_path, sample_idx=0, frame_idx=10):
    """
    从 .pkl 文件中读取真实的骨架数据
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # PYSKL 的标注通常是列表，每个元素是一个样本的字典
    # 或者是一个字典包含 'annotations' 列表
    if isinstance(data, dict) and 'annotations' in data:
        annotations = data['annotations']
    else:
        annotations = data

    sample = annotations[sample_idx]

    # 获取骨架坐标: 形状通常是 (M, T, V, C) -> (人数, 帧数, 关节数, 2)
    # 我们这里只取第一个人 (M=0) 的特定帧
    # 注意：真实数据的坐标是基于原图分辨率的 (例如 1920x1080)
    # PoseConv3D 训练时通常会 resize 到更小的尺寸 (如 56x56 的特征图)
    # 所以这里我们需要模拟一个缩放操作

    raw_kps = sample['keypoint'][0, frame_idx]  # 取第0个人，第frame_idx帧
    raw_scores = sample['keypoint_score'][0, frame_idx]

    orig_w, orig_h = sample.get('img_shape', (480, 640))[:2]  # 获取原图尺寸

    return raw_kps, raw_scores, (orig_w, orig_h)

def load_real_sample(pkl_path, sample_idx=0, frame_idx=None):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    annotations = data['annotations'] if isinstance(data, dict) else data
    sample = annotations[sample_idx]

    keypoint = sample['keypoint']        # (M, T, V, C)
    score = sample['keypoint_score']

    M, T, V, C = keypoint.shape
    print(f"Skeleton shape: M={M}, T={T}, V={V}, C={C}")

    # 如果没指定帧，或者越界，自动选中间帧
    if frame_idx is None or frame_idx >= T:
        frame_idx = T // 2

    raw_kps = keypoint[0, frame_idx]
    raw_scores = score[0, frame_idx]

    orig_w, orig_h = sample.get('img_shape', (1920, 1080))[:2]

    return raw_kps, raw_scores, (orig_w, orig_h)


# ==========================================
# 3. 主执行逻辑
# ==========================================
def run_real_experiment():
    # --- 配置 ---
    pkl_file = '/media/zjq/6CDE775EDE771F8E/lxk/video_rgbPose/fall_dataset/fall_dataset_skeleton.pkl'  # 你的真实文件路径
    target_h, target_w = 64, 64  # 目标特征图大小 (模拟 CNN 的输入)

    try:
        # 1. 加载真实数据
        # sample_idx=0 (第一个视频), frame_idx=20 (第20帧，假设这帧有人)
        real_kps, real_scores, orig_shape = load_real_sample(pkl_file, sample_idx=0)
        print("keypoint_score stats:")
        print("min:", real_scores.min())
        print("max:", real_scores.max())
        print("mean:", real_scores.mean())

        # 2. 坐标缩放 (关键步骤)
        # 因为热图是在小的特征图上生成的 (比如 56x56 或 64x64)，
        # 而原始坐标是基于 1920x1080 的，所以必须按比例缩小坐标
        scale_x = target_w / orig_shape[0]
        scale_y = target_h / orig_shape[1]

        scaled_kps = real_kps.copy()
        scaled_kps[:, 0] *= scale_x
        scaled_kps[:, 1] *= scale_y
        print(
            "scaled x:", scaled_kps[:, 0].min(), scaled_kps[:, 0].max(),
            "scaled y:", scaled_kps[:, 1].min(), scaled_kps[:, 1].max()
        )
        print(f"原始分辨率: {orig_shape}, 目标分辨率: ({target_w}, {target_h})")
        print(f"加载了 {len(scaled_kps)} 个关键点")

        # 3. 绘图对比
        sigmas = [0.3, 0.7, 1.5]
        titles = [
            r'(a) $\sigma=0.3$ (Too Sparse)',
            r'(b) $\sigma=0.7$ (Optimal)',
            r'(c) $\sigma=1.5$ (Blurry)'
        ]

        plt.figure(figsize=(15, 5))

        # ===== 在 for 循环前，加这一行 =====
        visual_scores = np.ones_like(real_scores)
        for i, sigma in enumerate(sigmas):
            heatmap = generate_gaussian_heatmap(
                target_h, target_w, scaled_kps, sigma,visual_scores )

            plt.subplot(1, 3, i + 1)
            plt.imshow(heatmap, cmap='hot', interpolation='nearest')
            plt.title(titles[i], fontsize=14)
            plt.axis('off')

        plt.tight_layout()
        plt.savefig('real_data_sigma_experiment.png', dpi=150)
        print("图表已保存为 real_data_sigma_experiment.png")
        plt.show()

    except FileNotFoundError:
        print(f"错误：找不到文件 {pkl_file}。请确保你在项目根目录下运行，或者修改路径。")
    except Exception as e:
        print(f"发生错误: {e}")
        print("提示：如果这是你自己生成的数据，请检查 .pkl 文件的结构是否包含 'keypoint' 字段。")


if __name__ == "__main__":
    run_real_experiment()