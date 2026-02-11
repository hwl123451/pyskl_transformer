from functools import partial
import numpy as np

from tqdm.auto import tqdm
import cv2
import torch

import detectron2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

import pytorchvideo
from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,
)
from torchvision.transforms._functional_video import normalize
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slowfast_r50_detection # Another option is slowfast_r50_detection

from visualization import VideoVisualizer

from src.body import Body
from src.hand import Hand
import src.util as util  # from src.util

import copy
import numpy as np
import cv2
import torch


body_estimation = Body('body_pose_model.pth')


def expand_bbox(box, scale=1.3, image_shape=None):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2, y1 + h / 2
    new_w, new_h = w * scale, h * scale
    new_x1 = max(int(cx - new_w / 2), 0)
    new_y1 = max(int(cy - new_h / 2), 0)
    new_x2 = min(int(cx + new_w / 2), image_shape[1] - 1)
    new_y2 = min(int(cy + new_h / 2), image_shape[0] - 1)
    return new_x1, new_y1, new_x2, new_y2

def draw_openpose_body_and_hand(frame_bgr, box):
    """box: [x1, y1, x2, y2]"""
    x1, y1, x2, y2 = expand_bbox(box, scale=1.3, image_shape=frame_bgr.shape)
    crop = frame_bgr[y1:y2, x1:x2].copy()

    candidate, subset = body_estimation(crop)
    print('len(candidate)=', len(candidate))
    canvas = util.draw_bodypose(crop, candidate, subset)

    canvas_resized = cv2.resize(canvas, (x2 - x1, y2 - y1))
    frame_bgr[y1:y2, x1:x2] = canvas_resized
    return frame_bgr


device = 'cuda' # or 'cpu'
video_model = slowfast_r50_detection(True) # Another option is slowfast_r50_detection
video_model = video_model.eval().to(device)


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.55  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)


# This method takes in an image and generates the bounding boxes for people in the image.
def get_person_bboxes(inp_img, predictor):
    predictions = predictor(inp_img.cpu().detach().numpy())['instances'].to('cpu')
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = np.array(predictions.pred_classes.tolist() if predictions.has("pred_classes") else None)
    predicted_boxes = boxes[np.logical_and(classes==0, scores>0.75 )].tensor.cpu() # only person
    return predicted_boxes




def infer_pose_single_crop(crop_bgr, box_id=0):
    """crop_bgr is a np.ndarray [H, W, 3] in BGR format"""
    img, img_pad, pad, scale = scale_and_crop(crop_bgr)
    img_tensor = normalize_tensor(img_pad)
    with torch.no_grad():
        paf, heatmap = pose_net(img_tensor.unsqueeze(0).to('cuda'))

    # Pose NMS
    humans = pose_nms(heatmap[0], paf[0])

    # Shift keypoints to original image position
    drawn = draw_bodypose(crop_bgr.copy(), humans)
    return drawn, humans


def draw_openpose_on_frame(frame_bgr, boxes):
    frame_out = frame_bgr.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = expand_bbox(box.int().tolist(), scale=1.3, image_shape=frame_bgr.shape)
        crop = frame_bgr[y1:y2, x1:x2]
        crop_out, humans = infer_pose_single_crop(crop)
        if len(humans) == 0:
            continue
        # Paste crop with drawn skeletons back
        frame_out[y1:y2, x1:x2] = crop_out
    return frame_out


def ava_inference_transform(
    clip,
    boxes,
    num_frames = 32, #if using slowfast_r50_detection, change this to 32
    crop_size = 256,
    data_mean = [0.45, 0.45, 0.45],
    data_std = [0.225, 0.225, 0.225],
    slow_fast_alpha = 4, #if using slowfast_r50_detection, change this to 4
):

    boxes = np.array(boxes)
    ori_boxes = boxes.copy()

    # Image [0, 255] -> [0, 1].
    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float()
    clip = clip / 255.0

    height, width = clip.shape[2], clip.shape[3]
    # The format of boxes is [x1, y1, x2, y2]. The input boxes are in the
    # range of [0, width] for x and [0,height] for y
    boxes = clip_boxes_to_image(boxes, height, width)

    # Resize short side to crop_size. Non-local and STRG uses 256.
    clip, boxes = short_side_scale_with_boxes(
        clip,
        size=crop_size,
        boxes=boxes,
    )

    # Normalize images by mean and std.
    clip = normalize(
        clip,
        np.array(data_mean, dtype=np.float32),
        np.array(data_std, dtype=np.float32),
    )

    boxes = clip_boxes_to_image(
        boxes, clip.shape[2],  clip.shape[3]
    )

    # Incase of slowfast, generate both pathways
    if slow_fast_alpha is not None:
        fast_pathway = clip
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            clip,
            1,
            torch.linspace(
                0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha
            ).long(),
        )
        clip = [slow_pathway, fast_pathway]

    return clip, torch.from_numpy(boxes), ori_boxes


# Create an id to label name mapping
label_map, allowed_class_ids = AvaLabeledVideoFramePaths.read_label_map('ava_action_list.pbtxt')
# Create a video visualizer that can plot bounding boxes and visualize actions on bboxes.
video_visualizer = VideoVisualizer(81, label_map, top_k=3, mode="thres",thres=0.5)

# video_path = '/home/yuting/video_processing/theatre.webm'
video_path = '/home/yuting/video_processing/-5KQ66BBWC4.mkv'
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps
print(f"Video FPS: {fps}, Total frames: {total_frames}, Duration: {duration:.2f}s")

time_stamp_range = range(0, min(int(duration), 100))  # limit to 100s
clip_duration = 1.0  # 1 second per inference
num_frames = 32      # for SlowFast

gif_imgs = []

def read_clip(cap, start_sec, end_sec, num_frames, resize=(256, 256)):
    """Read frames between start_sec and end_sec and return as tensor [C, T, H, W]"""
    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)
    frames = []
    interval = (end_sec - start_sec) / num_frames
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_MSEC, (start_sec + i * interval) * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        if resize:
            frame = cv2.resize(frame, resize)
        frame = torch.from_numpy(frame).permute(2, 0, 1)  # [C, H, W]
        frames.append(frame)
    if len(frames) < num_frames:
        return None
    return torch.stack(frames, dim=1)  # [C, T, H, W]


output_path = 'output.avi'
video_writer = None  # To be initialized after first valid frame

for time_stamp in tqdm(time_stamp_range):
    print(f"Generating predictions for time stamp: {time_stamp} sec")
    inp_imgs = read_clip(cap, time_stamp - clip_duration / 2, time_stamp + clip_duration / 2, num_frames)
    if inp_imgs is None:
        print(f"Skipping: insufficient frames at {time_stamp}")
        continue

    inp_img = inp_imgs[:, num_frames // 2, :, :].permute(1, 2, 0)  # middle frame [H, W, C]
    predicted_boxes = get_person_bboxes(inp_img, predictor)
    if len(predicted_boxes) == 0:
        print(f"Skipping: no people detected at {time_stamp}")
        continue

    inputs, inp_boxes, _ = ava_inference_transform(inp_imgs, predicted_boxes.numpy())
    inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0], 1), inp_boxes], dim=1)
    preds = video_model([x.unsqueeze(0).to(device) for x in inputs], inp_boxes.to(device))
    preds = torch.cat([torch.zeros(preds.shape[0], 1), preds.cpu()], dim=1)

    inp_imgs_np = inp_imgs.permute(1, 2, 3, 0).float() / 255.0  # [T, H, W, C]
    out_img_pred = video_visualizer.draw_clip_range(inp_imgs_np, preds, predicted_boxes)

    # Initialize writer on first valid frame
    if video_writer is None and len(out_img_pred) > 0:
        height, width = out_img_pred[0].shape[:2]
        video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'), 7, (width, height))

    for frame in out_img_pred:
        frame_bgr = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        for box in predicted_boxes:
            frame_bgr = draw_openpose_body_and_hand(frame_bgr, box.int().tolist())

        video_writer.write(frame_bgr)

cap.release()
if video_writer:
    video_writer.release()
    print(f'Predictions are saved to the video file: {output_path}')
else:
    print("No valid frames processed, no video saved.")
