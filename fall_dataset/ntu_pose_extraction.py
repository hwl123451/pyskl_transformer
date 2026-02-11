import os.path as osp
from collections import defaultdict
from tempfile import TemporaryDirectory

import torch
import numpy as np
import cv2
import mmcv
from mmcv import Config
from mmengine import dump

# ----------- DETECTION (MMDetection) ----------------
def load_detector(config_path, checkpoint_path, device='cuda:0'):
    from mmdet.models import build_detector
    cfg = Config.fromfile(config_path)
    cfg.model.pretrained = None
    detector = build_detector(cfg.model)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    detector.load_state_dict(checkpoint['state_dict'], strict=False)
    detector.to(device)
    detector.eval()
    return detector, cfg

def detect_person(detector, cfg, img, score_thr=0.5, device='cuda:0'):
    # MMDet image pipeline
    test_pipeline = mmcv.Compose(cfg.test_pipeline)
    data = dict(img=img, img_shape=img.shape, ori_shape=img.shape)
    data = test_pipeline(data)
    data = {k: v for k, v in data.items() if k not in ['img_meta']}
    with torch.no_grad():
        result = detector(data['img'].unsqueeze(0).to(device))[0]
    # COCO: person class is 0
    bboxes = result[0]  # person
    bboxes = bboxes[bboxes[:, 4] > score_thr]
    return bboxes.cpu().numpy()

def detection_inference(det_config, det_checkpoint, frame_paths, score_thr, device='cuda:0'):
    detector, cfg = load_detector(det_config, det_checkpoint, device)
    results = []
    for frame_path in frame_paths:
        img = mmcv.imread(frame_path)
        bboxes = detect_person(detector, cfg, img, score_thr, device)
        results.append(bboxes)
    return results, None

# ----------- POSE (MMPose) ----------------
def load_pose_model(config_path, checkpoint_path, device='cuda:0'):
    from mmpose.models import build_posenet
    cfg = Config.fromfile(config_path)
    cfg.model.pretrained = None
    pose_model = build_posenet(cfg.model)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    pose_model.load_state_dict(checkpoint['state_dict'], strict=False)
    pose_model.to(device)
    pose_model.eval()
    return pose_model, cfg

def single_pose_inference(pose_model, cfg, img, bbox, device='cuda:0'):
    # Crop and preprocess
    x1, y1, x2, y2, score = bbox
    person_img = img[int(y1):int(y2), int(x1):int(x2)]
    # apply preprocessing per config
    # ... (resize, normalize, pad, etc. from cfg.test_pipeline)
    # For demo: just resize to input_size
    input_size = cfg.data_cfg['image_size'] if 'image_size' in cfg.data_cfg else (192,256)
    person_img = cv2.resize(person_img, input_size)
    person_img = person_img.transpose(2,0,1)[None] / 255.0
    person_img = torch.from_numpy(person_img).float().to(device)
    with torch.no_grad():
        out = pose_model(person_img)
        # Assume output is (B, num_keypoints, 2)
        keypoints = out[0].cpu().numpy()
        scores = np.ones(keypoints.shape[0]) * score  # Dummy: propagate det score
    return keypoints, scores

def pose_inference(pose_config, pose_checkpoint, frame_paths, det_results, device='cuda:0'):
    pose_model, cfg = load_pose_model(pose_config, pose_checkpoint, device)
    pose_results = []
    for frame_path, bboxes in zip(frame_paths, det_results):
        img = mmcv.imread(frame_path)
        kpts = []
        scores = []
        for bbox in bboxes:
            kp, sc = single_pose_inference(pose_model, cfg, img, bbox, device)
            kpts.append(kp)
            scores.append(sc)
        pose_results.append({'keypoints': np.array(kpts), 'keypoint_scores': np.array(scores)})
    return pose_results, None

# ----------- FRAME EXTRACTION ----------------
def extract_frames(video_path, out_dir):
    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = osp.join(out_dir, f'frame_{frame_idx:05d}.jpg')
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        frame_idx += 1
    cap.release()
    return frame_paths, frame_idx

# ----------- REST LOGIC (postproc, etc.) ----------------
# (use your original utility functions: removedup, ntu_det_postproc, etc.)

# ----------- MAIN PROCESS (same as before) ----------------
def ntu_pose_extraction(vid, skip_postproc=False):
    tmp_dir = TemporaryDirectory()
    frame_paths, _ = extract_frames(vid, tmp_dir.name)
    det_results, _ = detection_inference(
        'faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py',
        'faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth',
        frame_paths,
        0.5,
        device='cuda:0')
    # ... your post-processing as before ...
    # pose inference
    pose_results, _ = pose_inference(
        'demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py',
        'hrnet_w32_coco_256x192-c78dce93_20200708.pth',
        frame_paths,
        det_results,
        device='cuda:0')
    # ... align, save anno as before ...
    tmp_dir.cleanup()
    return {'pose_results': pose_results}

# ----------- Example usage/test ----------------
if __name__ == '__main__':
    # Replace with argparse if needed
    vid = 'your_video.mp4'
    anno = ntu_pose_extraction(vid)
    dump(anno, 'out.pkl')