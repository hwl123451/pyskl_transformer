import os
import mmcv
from tqdm.auto import tqdm


# Modify these paths
annotation_path = '/root/shared-storage/pyskl/data/ntu60_hrnet.pkl'
video_root = '/root/shared-storage/pyskl/data/nturgbd_videos'
filtered_annotation_path = 'data/nturgbd/ntu60_hrnet_incomplete.pkl'


# Load original annotation
data = mmcv.load(annotation_path)

# Prepare sets for fast lookup
valid_frame_dirs = set()
filtered_annotations = []

# Step 1: Filter annotations based on existing .mp4 files
for sample in tqdm(data['annotations'], desc='Filter annotations based on existing .mp4 files'):
    video_path = os.path.join(video_root, sample['frame_dir'] + '.mp4')
    if os.path.exists(video_path):
        filtered_annotations.append(sample)
        valid_frame_dirs.add(sample['frame_dir'])

print(f"Filtered {len(data['annotations']) - len(filtered_annotations)} missing entries.")
data['annotations'] = filtered_annotations

# Step 2: Clean up each split list
for split_name in tqdm(data['split'], desc='Processing split'):
    before = len(data['split'][split_name])
    data['split'][split_name] = [
        name for name in data['split'][split_name] if name in valid_frame_dirs
    ]
    after = len(data['split'][split_name])
    print(f"Updated split '{split_name}': {before} → {after}")

# Save filtered .pkl
mmcv.dump(data, filtered_annotation_path)
print(f"Saved cleaned annotations to: {filtered_annotation_path}")
