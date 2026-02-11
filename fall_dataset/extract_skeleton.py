import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import random
#user to made skeleton  (train/val/test config) ======>pkl
def draw_skeleton(image, keypoints, scores, threshold=0.2, color=(0,255,0)):
    skeleton = [
        [5, 7], [7, 9],      # Left Arm
        [6, 8], [8, 10],     # Right Arm
        [11, 13], [13, 15],  # Left Leg
        [12, 14], [14, 16],  # Right Leg
        [5, 6],              # Shoulders
        [11, 12],            # Hips
        [5, 11], [6, 12],    # Spine
        [0, 1], [0, 2],      # Eyes
        [1, 3], [2, 4]       # Ears
    ]
    for i, (x, y) in enumerate(keypoints):
        if scores[i] > threshold:
            cv2.circle(image, (int(x), int(y)), 3, color, -1)
    for i, j in skeleton:
        if scores[i] > threshold and scores[j] > threshold:
            pt1 = (int(keypoints[i][0]), int(keypoints[j][0]))
            pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
            cv2.line(image, pt1, pt2, color, 2)
    return image

def process_video_png_dir(
    png_dir: str,
    label: int = 0,
    video_id: str = None,
    vis_dir: str = None,
    model=None
):
    images = sorted([f for f in os.listdir(png_dir) if f.lower().endswith('.png')])
    total_frames = len(images)
    if total_frames == 0:
        print(f"Warning: No PNG files found in {png_dir}. Skipping...")
        return None

    sample_img = cv2.imread(os.path.join(png_dir, images[0]))
    frame_shape = sample_img.shape[:2]  # (height, width)
    V = 17  # Number of keypoints for YOLOv8-pose
    C = 2   # x, y
    M = 1   # Number of persons (can extend to N later if needed)

    keypoint = np.zeros((total_frames, M, V, C), dtype=np.float32)
    keypoint_score = np.zeros((total_frames, M, V), dtype=np.float32)

    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)

    for t, img_name in enumerate(images):
        img_path = os.path.join(png_dir, img_name)
        image = cv2.imread(img_path)
        results = model(image)
        if len(results[0]) > 0:
            kp_xy = results[0].keypoints.xy[0].cpu().numpy()  # [17, 2]
            kp_conf = results[0].keypoints.conf[0].cpu().numpy()  # [17]
            keypoint[t, 0] = kp_xy
            keypoint_score[t, 0] = kp_conf
            if vis_dir is not None:
                vis_img = image.copy()
                vis_img = draw_skeleton(vis_img, kp_xy, kp_conf)
                cv2.imwrite(os.path.join(vis_dir, img_name), vis_img)
        else:
            keypoint[t, 0] = np.zeros((V, C))
            keypoint_score[t, 0] = np.zeros(V)
            if vis_dir is not None:
                cv2.imwrite(os.path.join(vis_dir, img_name), image)  # No skeleton

    annotation = {
        'frame_dir': video_id if video_id else os.path.basename(png_dir),
        # Choose label based on directory name
        'label': 0 if 'adl' in png_dir else 1,
        'total_frames': total_frames,
        'keypoint': keypoint,          # shape: [T, M, V, C]
        'keypoint_score': keypoint_score, # shape: [T, M, V]
        'img_shape': frame_shape,      # [height, width]
        'start_index': 0
    }
    return annotation

def split_annotations(annotations, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    random.seed(seed)
    n = len(annotations)
    indices = list(range(n))
    random.shuffle(indices)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    train_ids = indices[:train_end]
    val_ids = indices[train_end:val_end]
    test_ids = indices[val_end:]
    return train_ids, val_ids, test_ids

def process_basedir_and_generate_pkl(
    base_dir: str,
    output_pkl: str,
    vis_dir: str = None,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
):
    # Find all subdirectories in base_dir (including nested ones)
    subdirs = sorted([
        dirpath for dirpath, subdirs, _ in os.walk(base_dir) if len(subdirs) == 0
    ])
    model = YOLO('yolov8n-pose.pt')
    all_annotations = []
    for subdir in tqdm(subdirs, desc="Processing videos"):
        video_id = os.path.basename(subdir)
        vis_subdir = os.path.join(vis_dir, video_id) if vis_dir is not None else None
        annotation = process_video_png_dir(
            png_dir=subdir,
            video_id=video_id,
            vis_dir=vis_subdir,
            model=model
        )
        if annotation is not None:
            all_annotations.append(annotation)
    # Split the dataset
    train_ids, val_ids, test_ids = split_annotations(all_annotations, train_ratio, val_ratio, test_ratio)

    # Prepare splits for config
    split_dict = {
        'xsub_train': [all_annotations[i]['frame_dir'] for i in train_ids],
        'xsub_val': [all_annotations[i]['frame_dir'] for i in val_ids],
        'xsub_test': [all_annotations[i]['frame_dir'] for i in test_ids],
    }
    # Save as dict with 'annotations' and 'split'
    out_dict = {
        'annotations': all_annotations,
        'split': split_dict
    }
    with open(output_pkl, 'wb') as f:
        pickle.dump(out_dict, f)
    print(f"Saved {len(all_annotations)} annotations with splits to {output_pkl}")
    print(f"Train: {len(split_dict['xsub_train'])}, Val: {len(split_dict['xsub_val'])}, Test: {len(split_dict['xsub_test'])}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Extract skeleton keypoints from PNG videos in a base directory, and format splits.')
    parser.add_argument('base_dir', type=str, help='Base directory containing multiple video subdirectories')
    parser.add_argument('output_pkl', type=str, help='Output pickle file for all skeleton annotations with splits')
    parser.add_argument('--vis_dir', type=str, default=None, help='Directory to save visualized skeleton images (per video)')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Training split ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation split ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test split ratio')
    args = parser.parse_args()

    process_basedir_and_generate_pkl(
        base_dir=args.base_dir,
        output_pkl=args.output_pkl,
        vis_dir=args.vis_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
