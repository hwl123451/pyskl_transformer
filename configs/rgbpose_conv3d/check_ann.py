import os
import glob


def check_ntu_data(data_root):
    print(f"正在检查目录: {data_root}")

    if not os.path.exists(data_root):
        print("❌ 错误: 目录不存在！请检查路径是否正确。")
        return

    # 获取所有 avi 文件
    video_files = glob.glob(os.path.join(data_root, '*.avi'))
    count = len(video_files)

    print(f"找到 .avi 文件数量: {count}")

    # NTU RGB+D 60 标准数量
    EXPECTED_COUNT = 56880

    if count == EXPECTED_COUNT:
        print("✅ 数量正确: 56880 个文件。")
    elif count == 0:
        print("❌ 错误: 该目录下没有 .avi 文件。")
        print("   提示: NTU 数据集原始格式是 .avi。如果你已经压缩过视频，可能是 .mp4。")
        print("   如果是 .mp4，请修改 config 中的 dataset 扩展名设置。")
        return
    else:
        print(f"⚠️ 警告: 文件数量 ({count}) 与 NTU60 标准 ({EXPECTED_COUNT}) 不一致。")
        print("   这可能是因为你使用的是 NTU120，或者数据不完整/经过了筛选。")

    # 检查命名格式 (S001C001P001R001A001.avi)
    sample_file = os.path.basename(video_files[0])
    if len(sample_file) == 24 and sample_file.startswith('S') and sample_file.endswith('.avi'):
        print(f"✅ 文件名格式看起来正确 (示例: {sample_file})")
    else:
        print(f"⚠️ 警告: 文件名格式可能不标准 (示例: {sample_file})")


if __name__ == "__main__":
    # 你的数据路径
    DATA_ROOT = '/media/zjq/6CDE775EDE771F8E/lxk/NTU-data/nturgbd_videos_all/'
    check_ntu_data(DATA_ROOT)