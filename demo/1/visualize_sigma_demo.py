import numpy as np
import matplotlib.pyplot as plt


def generate_gaussian_heatmap(img_h, img_w, centers, sigma, max_values):
    """
    独立实现的热图生成函数，对应 heatmap_related.py 中的逻辑
    """
    # 初始化全零热图
    arr = np.zeros((img_h, img_w), dtype=np.float32)

    for center, max_value in zip(centers, max_values):
        # 对应代码中的 EPS 过滤
        if max_value < 1e-3:
            continue

        mu_x, mu_y = center[0], center[1]

        # 确定高斯核的生成范围 (3*sigma 原则)
        st_x = max(int(mu_x - 3 * sigma), 0)
        ed_x = min(int(mu_x + 3 * sigma) + 1, img_w)
        st_y = max(int(mu_y - 3 * sigma), 0)
        ed_y = min(int(mu_y + 3 * sigma) + 1, img_h)

        # 生成网格坐标
        x = np.arange(st_x, ed_x, 1, np.float32)
        y = np.arange(st_y, ed_y, 1, np.float32)

        if not (len(x) and len(y)):
            continue

        y = y[:, None]

        # 核心公式 (3-2): Gaussian patch generation
        # patch = c * exp( - ||p - v||^2 / 2sigma^2 )
        patch = np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / 2 / sigma ** 2)
        patch = patch * max_value

        # 将 patch 叠加到热图上 (取最大值)
        arr[st_y:ed_y, st_x:ed_x] = np.maximum(arr[st_y:ed_y, st_x:ed_x], patch)

    return arr


def run_experiment():
    # 1. 实验设置
    # 特征图大小 (通常 PoseConv3D 使用 56x56 或类似的较小分辨率)
    H, W = 64, 64

    # 模拟两个相邻的关键点 (例如：手肘和手腕)
    # 距离设置为 5 个像素，用于测试 sigma 过大时的粘连情况
    keypoints = np.array([
        [32, 28],  # 关键点 A
        [32, 33]  # 关键点 B (距离 A 5个像素)
    ])

    # 模拟置信度 (均为 1.0)
    scores = np.array([1.0, 1.0])

    # 定义需要对比的三个 sigma 值
    sigmas = [0.3, 0.7, 1.5]
    titles = [
        r'(a) $\sigma=0.3$ (Too Sparse)',
        r'(b) $\sigma=0.7$ (Optimal)',
        r'(c) $\sigma=1.5$ (Blurry/Merged)'
    ]

    # 2. 生成并画图
    plt.figure(figsize=(15, 5))

    for i, sigma in enumerate(sigmas):
        # 调用生成函数
        heatmap = generate_gaussian_heatmap(H, W, keypoints, sigma, scores)

        # 绘图
        plt.subplot(1, 3, i + 1)
        plt.imshow(heatmap, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
        plt.title(titles[i], fontsize=14)
        plt.axis('off')

        # 添加一些文字说明
        if sigma == 0.3:
            plt.text(H // 2, W + 5, "Features are too sparse\nGradients hard to propagate",
                     ha='center', va='top', fontsize=10, color='blue')
        elif sigma == 0.7:
            plt.text(H // 2, W + 5, "Clear peaks, smooth edges\nGood spatial support",
                     ha='center', va='top', fontsize=10, color='green')
        elif sigma == 1.5:
            plt.text(H // 2, W + 5, "Peaks merge together\nStructure lost",
                     ha='center', va='top', fontsize=10, color='red')

    plt.tight_layout()

    # 保存结果
    plt.savefig('sigma_comparison_experiment.png', dpi=300, bbox_inches='tight')
    print("实验结果已保存为 'sigma_comparison_experiment.png'")
    plt.show()


if __name__ == "__main__":
    run_experiment()