# # Copyright (c) OpenMMLab. All rights reserved.
# import os.path as osp
#
# from .base import BaseDataset
# from .builder import DATASETS
#
#
# @DATASETS.register_module()
# class VideoDataset(BaseDataset):
#     """Video dataset for action recognition.
#
#     The dataset loads raw videos and apply specified transforms to return a
#     dict containing the frame tensors and other information.
#
#     The ann_file is a text file with multiple lines, and each line indicates
#     a sample video with the filepath and label, which are split with a
#     whitespace. Example of a annotation file:
#
#     .. code-block:: txt
#
#         some/path/000.mp4 1
#         some/path/001.mp4 1
#         some/path/002.mp4 2
#         some/path/003.mp4 2
#         some/path/004.mp4 3
#         some/path/005.mp4 3
#
#
#     Args:
#         ann_file (str): Path to the annotation file.
#         pipeline (list[dict | callable]): A sequence of data transforms.
#         start_index (int): Specify a start index for frames in consideration of
#             different filename format. However, when taking videos as input,
#             it should be set to 0, since frames loaded from videos count
#             from 0. Default: 0.
#         **kwargs: Keyword arguments for ``BaseDataset``.
#     """
#
#     def __init__(self, ann_file, pipeline, start_index=0, **kwargs):
#         super().__init__(ann_file, pipeline, start_index=start_index, **kwargs)
#
#     def load_annotations(self):
#         """Load annotation file to get video information."""
#         if self.ann_file.endswith('.json'):
#             return self.load_json_annotations()
#
#         video_infos = []
#         with open(self.ann_file, 'r') as fin:
#             for line in fin:
#                 line_split = line.strip().split()
#                 if self.multi_class:
#                     assert self.num_classes is not None
#                     filename, label = line_split[0], line_split[1:]
#                     label = list(map(int, label))
#                 else:
#                     filename, label = line_split
#                     label = int(label)
#                 filename = osp.join(self.data_prefix, filename)
#                 video_infos.append(dict(filename=filename, label=label))
#         return video_infos

#use only rgb next
# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import mmcv
from .base import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class VideoDataset(BaseDataset):
    """Video dataset for action recognition.

    Modified to support loading from .pkl files with 'split' argument,
    and forcing .avi extension for NTU dataset.
    """

    def __init__(self, ann_file, pipeline, start_index=0, split=None, **kwargs):
        self.split = split
        super().__init__(ann_file, pipeline, start_index=start_index, **kwargs)

    def load_annotations(self):
        if self.ann_file.endswith('.json'):
            return self.load_json_annotations()
        elif self.ann_file.endswith('.pkl'):
            return self.load_pkl_annotations()

        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                if self.multi_class:
                    assert self.num_classes is not None
                    filename, label = line_split[0], line_split[1:]
                    label = list(map(int, label))
                else:
                    filename, label = line_split
                    label = int(label)
                filename = osp.join(self.data_prefix, filename)
                video_infos.append(dict(filename=filename, label=label))
        return video_infos

    def load_pkl_annotations(self):
        data = mmcv.load(self.ann_file)

        if self.split:
            split, data = data['split'], data['annotations']
            identifier = 'filename' if 'filename' in data[0] else 'frame_dir'
            split = set(split[self.split])
            data = [x for x in data if x[identifier] in split]

        for item in data:
            # 1. 获取原始文件名 (可能是 frame_dir 或 filename)
            # 某些 pkl 可能带路径前缀 (如 'fall_rgb/S001...')，我们需要去掉它
            raw_name = item.get('filename', item.get('frame_dir', ''))

            # 只取文件名部分 (例如 'S001C001P001R001A001')
            video_id = osp.basename(raw_name)

            # 去掉可能的后缀 (防止出现 .mp4.avi)
            if '.' in video_id:
                video_id = os.path.splitext(video_id)[0]

            # 2. 强制加上 .avi 后缀 (因为你的数据是 avi)
            new_filename = video_id + '.avi'

            # 3. 拼接完整绝对路径
            item['filename'] = osp.join(self.data_prefix, new_filename)

            # 4. [关键] 将 frame_dir 设置为同样的值或删除
            # 这样可以防止 MMDecode 里的逻辑去寻找不存在的文件夹
            item['frame_dir'] = item['filename']

        return data