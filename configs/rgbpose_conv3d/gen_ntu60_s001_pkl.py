import mmengine
import os
import random
import numpy as np


def generate_subset_pkl():
    # 1. 路径配置 (请修改为你实际的路径)
    # 官方完整的 NTU60 HRNet keypoints 文件路径
    full_pkl_path = '/media/zjq/6CDE775EDE771F8E/lxk/NTU-data/ntu60_hrnet.pkl'
    # 你的 S001 视频所在的文件夹 (用于检查文件名后缀)
    video_folder = '/media/zjq/6CDE775EDE771F8E/lxk/NTU-data/nturgbd_raw/'

    # 输出文件路径
    out_train_path = '/media/zjq/6CDE775EDE771F8E/lxk/NTU-data/ntu60_s001_train.pkl'
    out_val_path = '/media/zjq/6CDE775EDE771F8E/lxk/NTU-data/ntu60_s001_val.pkl'

    print(f"Loading full annotations from {full_pkl_path}...")
    try:
        data = mmengine.load(full_pkl_path)
    except FileNotFoundError:
        print("错误：找不到完整的 ntu60_hrnet.pkl 文件，请先下载官方提供的骨骼数据。")
        return

    annotations = data['annotations']
    print(f"Total annotations in full dataset: {len(annotations)}")

    # 2. 筛选 S001 的数据
    s001_annotations = []

    # 获取文件夹里实际的视频文件名列表，用于比对
    # 注意：NTU文件名格式通常为 S001C001P001R001A001_rgb.avi 或 S001C001P001R001A001.avi
    # 这里的逻辑是只要包含 S001 且在pkl里存在的就保留

    valid_count = 0
    for item in annotations:
        frame_dir = item['frame_dir']  # 通常是 S001C001P001R001A001

        # 检查是否属于 S001 Setup
        if "S001" in frame_dir:
            # 修正文件名后缀问题
            # 如果你的视频文件带 .avi，但 pkl 里的 frame_dir 没有后缀，
            # 需要确保 Config 里的 Pipeline 能处理，或者在这里给 frame_dir 加上后缀。
            # 为了保险，我们假设 pyskl 使用不带后缀的 ID 去匹配，或者在 MMDecode 里处理。
            # 如果训练时报错找不到文件，可以尝试在这里加上: item['frame_dir'] = frame_dir + '.avi'

            s001_annotations.append(item)

    print(f"Found {len(s001_annotations)} videos for S001.")

    if len(s001_annotations) == 0:
        print("Warning: 没有找到 S001 的数据，请检查 pkl 文件内容格式。")
        return

    # 3. 划分训练集和验证集 (例如 8:2)
    # 因为只有一个 Setup，没法用 Cross-Subject，我们随机划分
    random.seed(42)
    random.shuffle(s001_annotations)

    split_idx = int(len(s001_annotations) * 0.8)
    train_data = s001_annotations[:split_idx]
    val_data = s001_annotations[split_idx:]

    print(f"Split result -> Train: {len(train_data)}, Val: {len(val_data)}")

    # 4. 保存为 pyskl 可识别的格式
    # pyskl 的 PoseDataset 通常直接读取列表，或者包含 'annotations' 键的字典
    # 为了保险，我们保存为列表结构 (pyskl 通用结构)

    # 注意：如果原始 data 是 list，直接保存 list。如果是 dict 且包含 split 信息，我们只保存 annotations list。
    # 根据 pyskl 源码，PoseDataset 读取的是 annotations 列表。

    mmengine.dump(train_data, out_train_path)
    mmengine.dump(val_data, out_val_path)

    print(f"Success! Files saved to:\n{out_train_path}\n{out_val_path}")


if __name__ == '__main__':
    generate_subset_pkl()