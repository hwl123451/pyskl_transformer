# import torch
# import torch.nn as nn
# from mmcv.cnn import constant_init, kaiming_init
# from mmcv.runner import load_checkpoint
# from mmcv.utils import _BatchNorm, print_log
#
# from ...utils import get_root_logger
# from ..builder import BACKBONES
# from .resnet3d_slowfast import ResNet3dPathway
#
# from torchvision import models
# import math
# from torchvision.models.vision_transformer import Encoder
#
# class PatchEmbedding(nn.Module):
#     def __init__(self, d_model: int, learnable=False, dropout: float = 0.1, max_len: int = 15):
#         super(PatchEmbedding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
#         frames = max_len + 1
#         if learnable:
#             self.pe = nn.Parameter(torch.randn((1, frames, d_model)))
#         else:
#             position = torch.arange(frames).unsqueeze(1)
#             div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#             pe = torch.zeros(1, frames, d_model)
#             pe[0, :, 0::2] = torch.sin(position * div_term)
#             pe[0, :, 1::2] = torch.cos(position * div_term)
#             self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         b, _, _ = x.shape
#         cls_tokens = self.cls_token.expand(b, -1, -1)
#         x = torch.cat([cls_tokens, x], dim=1)
#         x = x + self.pe[:, :x.size(1), :]
#         return self.dropout(x)
#
#
# class FeatureExtractor(nn.Module):
#     def __init__(self, model_name: str, model_weights: str = 'DEFAULT'):
#         super(FeatureExtractor, self).__init__()
#         # Try models with `weights` argument
#         try:
#             self.model = getattr(models, model_name)(weights=model_weights)
#         except TypeError:
#             # Fallback to models with `pretrained` argument (older API)
#             self.model = getattr(models, model_name)(pretrained=(model_weights != None and model_weights != 'DEFAULT'))
#
#         self.d_model = None
#
#         # EfficientNet/ConvNeXT
#         if model_name in ['efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l', 'convnext_base',
#                           'convnext_small', 'convnext_large']:
#             self.d_model = self.model.classifier[-1].in_features
#             self.model.classifier[-1] = nn.Identity()
#         else:
#             self.d_model = self.model.fc.in_features
#             self.model.fc = nn.Identity()
#
#     def forward(self, x):
#         x = x.permute(0, 2, 1, 3, 4)
#         b, f, _, _, _ = x.shape
#         # Use reshape instead of view
#         x = x.reshape(b * f, *x.size()[2:])
#         x = self.model(x)
#         x = x.reshape(b, f, *x.size()[1:])
#         return x
#
# @BACKBONES.register_module()
# class ConvAcTransformer(nn.Module):
#     def __init__(self,
#                  attention_heads: int,
#                  num_layers: int,
#                  num_classes: int,
#                  num_frames: int,
#                  drop_p: float,
#                  feature_extractor_name: str,
#                  learnable_pe: bool = False):
#         super(ConvAcTransformer, self).__init__()
#         self.feature_extractor_name = feature_extractor_name
#         self.attention_heads = attention_heads
#         self.num_classes = num_classes
#         self.num_layers = num_layers
#         self.num_frames = num_frames
#         self.learnable_pe = learnable_pe
#         self.drop_p = drop_p
#
#         self.feature_extract = FeatureExtractor(self.feature_extractor_name, model_weights='DEFAULT')
#         self.patch_embed = PatchEmbedding(self.feature_extract.d_model, learnable=self.learnable_pe,
#                                           max_len=self.num_frames, dropout=self.drop_p)
#
#         self.transformer_encoder = Encoder(seq_length=self.num_frames + 1,
#                                            num_layers=self.num_layers,
#                                            num_heads=self.attention_heads,
#                                            hidden_dim=self.feature_extract.d_model,
#                                            mlp_dim=self.feature_extract.d_model,
#                                            dropout=self.drop_p, attention_dropout=self.drop_p)
#
#         self.dropout = nn.Dropout(self.drop_p)
#         self.classification_head = nn.Linear(self.feature_extract.d_model, self.num_classes)
#
#     def forward(self, x):
#         x = self.feature_extract(x)
#         x = self.patch_embed(x)
#         x = self.transformer_encoder(x)
#         x = x[:, 0, :]
#         return x
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 kaiming_init(m)
#             elif isinstance(m, _BatchNorm):
#                 constant_init(m, 1)
#         # Note: Add custom weight init for Transformer if needed"
#
# @BACKBONES.register_module()
# class RGBPoseConv3DTransformer(nn.Module):
#     """Slowfast backbone with replaced RGB pathway as ConvAcTransformer."""
#     def __init__(self,
#                  pretrained=None,
#                  speed_ratio=4,
#                  channel_ratio=4,
#                  rgb_detach=False,
#                  pose_detach=False,
#                  rgb_drop_path=0,
#                  pose_drop_path=0,
#                  # New keys for ConvAcTransformer config:
#                  rgb_pathway=dict(
#                      attention_heads=4,
#                      num_layers=4,
#                      num_classes=101,
#                      num_frames=50,
#                      drop_p=0.1,
#                      feature_extractor_name='wide_resnet50_2',
#                      learnable_pe=False
#                  ),
#                  pose_pathway=dict(
#                      num_stages=3,
#                      stage_blocks=(4, 6, 3),
#                      lateral=True,
#                      lateral_inv=True,
#                      lateral_infl=16,
#                      lateral_activate=(0, 1, 1),
#                      in_channels=17,
#                      base_channels=32,
#                      out_indices=(2, ),
#                      conv1_kernel=(1, 7, 7),
#                      conv1_stride=(1, 1),
#                      pool1_stride=(1, 1),
#                      inflate=(0, 1, 1),
#                      spatial_strides=(2, 2, 2),
#                      temporal_strides=(1, 1, 1))):
#
#         super().__init__()
#         self.pretrained = pretrained
#         self.speed_ratio = speed_ratio
#         self.channel_ratio = channel_ratio
#
#         # ConvAcTransformer replaces ResNet3dPathway for RGB
#         self.rgb_path = ConvAcTransformer(**rgb_pathway)
#         # Pose pathway remains unchanged
#         self.pose_path = ResNet3dPathway(**pose_pathway)
#         self.rgb_detach = rgb_detach
#         self.pose_detach = pose_detach
#         assert 0 <= rgb_drop_path <= 1
#         assert 0 <= pose_drop_path <= 1
#         self.rgb_drop_path = rgb_drop_path
#         self.pose_drop_path = pose_drop_path
#
#     def init_weights(self):
#         """Initiate the parameters either from existing checkpoint or from scratch."""
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 kaiming_init(m)
#             elif isinstance(m, _BatchNorm):
#                 constant_init(m, 1)
#
#         if isinstance(self.pretrained, str):
#             logger = get_root_logger()
#             msg = f'load model from: {self.pretrained}'
#             print_log(msg, logger=logger)
#             load_checkpoint(self, self.pretrained, strict=True, logger=logger)
#         elif self.pretrained is None:
#             # Init two branch separately.
#             # For ConvAcTransformer, custom weights init (if needed) can be added here.
#             self.pose_path.init_weights()
#         else:
#             raise TypeError('pretrained must be a str or None')
#
#     def forward(self, imgs, heatmap_imgs):
#         """Defines the computation performed at every call.
#
#         Args:
#             imgs (torch.Tensor): The input data for RGB pathway (B, F, C, H, W).
#             heatmap_imgs (torch.Tensor): The input data for pose pathway.
#
#         Returns:
#             tuple[torch.Tensor]: The feature of the input samples extracted by the backbone.
#         """
#         if self.training:
#             rgb_drop_path = torch.rand(1) < self.rgb_drop_path
#             pose_drop_path = torch.rand(1) < self.pose_drop_path
#         else:
#             rgb_drop_path, pose_drop_path = False, False
#
#         # RGB path output is feature/class logits
#         x_rgb = self.rgb_path(imgs)  # (B, num_classes)
#
#         # Pose pathway as before
#         x_pose = self.pose_path.conv1(heatmap_imgs)
#         x_pose = self.pose_path.maxpool(x_pose)
#         x_pose = self.pose_path.layer1(x_pose)
#
#         if hasattr(self.pose_path, 'layer1_lateral'):
#             feat = x_rgb.detach() if self.pose_detach else x_rgb
#             x_rgb_lateral = self.pose_path.layer1_lateral(feat)
#             if pose_drop_path:
#                 x_rgb_lateral = x_rgb_lateral.new_zeros(x_rgb_lateral.shape)
#             x_pose = torch.cat((x_pose, x_rgb_lateral), dim=1)
#
#         x_pose = self.pose_path.layer2(x_pose)
#         if hasattr(self.pose_path, 'layer2_lateral'):
#             feat = x_rgb.detach() if self.pose_detach else x_rgb
#             x_rgb_lateral = self.pose_path.layer2_lateral(feat)
#             if pose_drop_path:
#                 x_rgb_lateral = x_rgb_lateral.new_zeros(x_rgb_lateral.shape)
#             x_pose = torch.cat((x_pose, x_rgb_lateral), dim=1)
#
#         x_pose = self.pose_path.layer3(x_pose)
#
#         return (x_rgb, x_pose)
#
#     def train(self, mode=True):
#         super().train(mode)
#         self.training = True
#
#     def eval(self):
#         super().eval()
#         self.training = False
#
# import torch
# import torch.nn as nn
# from mmcv.cnn import constant_init, kaiming_init
# from mmcv.runner import load_checkpoint
# from mmcv.utils import _BatchNorm, print_log
#
# from ...utils import get_root_logger
# from ..builder import BACKBONES
# from .resnet3d_slowfast import ResNet3dPathway
#
# from torchvision import models
# import math
# from torchvision.models.vision_transformer import Encoder
#
#
# class PatchEmbedding(nn.Module):
#     def __init__(self, d_model: int, learnable=False, dropout: float = 0.1, max_len: int = 15):
#         super(PatchEmbedding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
#         frames = max_len + 1
#         if learnable:
#             self.pe = nn.Parameter(torch.randn((1, frames, d_model)))
#         else:
#             position = torch.arange(frames).unsqueeze(1)
#             div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#             pe = torch.zeros(1, frames, d_model)
#             pe[0, :, 0::2] = torch.sin(position * div_term)
#             pe[0, :, 1::2] = torch.cos(position * div_term)
#             self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         b, _, _ = x.shape
#         cls_tokens = self.cls_token.expand(b, -1, -1)
#         x = torch.cat([cls_tokens, x], dim=1)
#         x = x + self.pe[:, :x.size(1), :]
#         return self.dropout(x)
#
#
# class FeatureExtractor(nn.Module):
#     def __init__(self, model_name: str, model_weights: str = 'DEFAULT'):
#         super(FeatureExtractor, self).__init__()
#         # Try models with `weights` argument
#         try:
#             self.model = getattr(models, model_name)(weights=model_weights)
#         except TypeError:
#             # Fallback to models with `pretrained` argument (older API)
#             self.model = getattr(models, model_name)(pretrained=(model_weights != None and model_weights != 'DEFAULT'))
#
#         self.d_model = None
#
#         # EfficientNet/ConvNeXT
#         if model_name in ['efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l', 'convnext_base',
#                           'convnext_small', 'convnext_large']:
#             self.d_model = self.model.classifier[-1].in_features
#             self.model.classifier[-1] = nn.Identity()
#         else:
#             self.d_model = self.model.fc.in_features
#             self.model.fc = nn.Identity()
#
#     def forward(self, x):
#         x = x.permute(0, 2, 1, 3, 4)
#         b, f, _, _, _ = x.shape
#         # Use reshape instead of view
#         x = x.reshape(b * f, *x.size()[2:])
#         x = self.model(x)
#         x = x.reshape(b, f, *x.size()[1:])
#         return x
#
#
# @BACKBONES.register_module()
# class ConvAcTransformer(nn.Module):
#     def __init__(self,
#                  attention_heads: int,
#                  num_layers: int,
#                  num_classes: int,
#                  num_frames: int,
#                  drop_p: float,
#                  feature_extractor_name: str,
#                  learnable_pe: bool = False):
#         super(ConvAcTransformer, self).__init__()
#         self.feature_extractor_name = feature_extractor_name
#         self.attention_heads = attention_heads
#         self.num_classes = num_classes
#         self.num_layers = num_layers
#         self.num_frames = num_frames
#         self.learnable_pe = learnable_pe
#         self.drop_p = drop_p
#
#         self.feature_extract = FeatureExtractor(self.feature_extractor_name, model_weights='DEFAULT')
#         self.patch_embed = PatchEmbedding(self.feature_extract.d_model, learnable=self.learnable_pe,
#                                           max_len=self.num_frames, dropout=self.drop_p)
#
#         self.transformer_encoder = Encoder(seq_length=self.num_frames + 1,
#                                            num_layers=self.num_layers,
#                                            num_heads=self.attention_heads,
#                                            hidden_dim=self.feature_extract.d_model,
#                                            mlp_dim=self.feature_extract.d_model,
#                                            dropout=self.drop_p, attention_dropout=self.drop_p)
#
#         self.dropout = nn.Dropout(self.drop_p)
#         # 注意：这个分类头在 backbone 里实际上没用，但为了保持结构一致先保留
#         self.classification_head = nn.Linear(self.feature_extract.d_model, self.num_classes)
#
#     def forward(self, x):
#         x = self.feature_extract(x)
#         x = self.patch_embed(x)
#         x = self.transformer_encoder(x)
#
#         # [修改点 1] 获取 CLS Token
#         cls_token = x[:, 0, :]  # (B, C)
#
#         # [修改点 2] 升维成 5D 张量 (B, C, 1, 1, 1)
#         # 这是为了兼容 I3DHead 以及后续 Fusion 时方便做广播
#         output = cls_token.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#
#         return output
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 kaiming_init(m)
#             elif isinstance(m, _BatchNorm):
#                 constant_init(m, 1)
#         # Note: Add custom weight init for Transformer if needed"
#
#
# @BACKBONES.register_module()
# class RGBPoseConv3DTransformer(nn.Module):
#     """Slowfast backbone with replaced RGB pathway as ConvAcTransformer."""
#
#     def __init__(self,
#                  pretrained=None,
#                  speed_ratio=4,
#                  channel_ratio=4,
#                  rgb_detach=False,
#                  pose_detach=False,
#                  rgb_drop_path=0,
#                  pose_drop_path=0,
#                  # New keys for ConvAcTransformer config:
#                  rgb_pathway=dict(
#                      attention_heads=4,
#                      num_layers=4,
#                      num_classes=101,
#                      num_frames=50,
#                      drop_p=0.1,
#                      feature_extractor_name='wide_resnet50_2',
#                      learnable_pe=False
#                  ),
#                  pose_pathway=dict(
#                      num_stages=3,
#                      stage_blocks=(4, 6, 3),
#                      lateral=True,
#                      lateral_inv=True,
#                      lateral_infl=16,
#                      lateral_activate=(0, 1, 1),
#                      in_channels=17,
#                      base_channels=32,
#                      out_indices=(2,),
#                      conv1_kernel=(1, 7, 7),
#                      conv1_stride=(1, 1),
#                      pool1_stride=(1, 1),
#                      inflate=(0, 1, 1),
#                      spatial_strides=(2, 2, 2),
#                      temporal_strides=(1, 1, 1))):
#
#         super().__init__()
#         self.pretrained = pretrained
#         self.speed_ratio = speed_ratio
#         self.channel_ratio = channel_ratio
#
#         # ConvAcTransformer replaces ResNet3dPathway for RGB
#         self.rgb_path = ConvAcTransformer(**rgb_pathway)
#         # Pose pathway remains unchanged
#         self.pose_path = ResNet3dPathway(**pose_pathway)
#         self.rgb_detach = rgb_detach
#         self.pose_detach = pose_detach
#         assert 0 <= rgb_drop_path <= 1
#         assert 0 <= pose_drop_path <= 1
#         self.rgb_drop_path = rgb_drop_path
#         self.pose_drop_path = pose_drop_path
#
#     def init_weights(self):
#         """Initiate the parameters either from existing checkpoint or from scratch."""
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 kaiming_init(m)
#             elif isinstance(m, _BatchNorm):
#                 constant_init(m, 1)
#
#         if isinstance(self.pretrained, str):
#             logger = get_root_logger()
#             msg = f'load model from: {self.pretrained}'
#             print_log(msg, logger=logger)
#             load_checkpoint(self, self.pretrained, strict=True, logger=logger)
#         elif self.pretrained is None:
#             # Init two branch separately.
#             self.pose_path.init_weights()
#             self.rgb_path.init_weights()  # 确保 RGB 分支也初始化
#         else:
#             raise TypeError('pretrained must be a str or None')
#
#     def forward(self, imgs, heatmap_imgs):
#         """Defines the computation performed at every call."""
#         if self.training:
#             rgb_drop_path = torch.rand(1) < self.rgb_drop_path
#             pose_drop_path = torch.rand(1) < self.pose_drop_path
#         else:
#             rgb_drop_path, pose_drop_path = False, False
#
#         # 1. RGB 分支前向 (B, C, 1, 1, 1)
#         x_rgb = self.rgb_path(imgs)
#
#         # 2. Pose 分支
#         x_pose = self.pose_path.conv1(heatmap_imgs)
#         x_pose = self.pose_path.maxpool(x_pose)
#
#         # Layer 1
#         x_pose = self.pose_path.layer1(x_pose)
#         if hasattr(self.pose_path, 'layer1_lateral'):
#             feat = x_rgb.detach() if self.pose_detach else x_rgb
#             x_rgb_lateral = self.pose_path.layer1_lateral(feat)  # (B, C_out, 1, 1, 1)
#
#             if pose_drop_path:
#                 x_rgb_lateral = x_rgb_lateral.new_zeros(x_rgb_lateral.shape)
#
#             # [修改点 3] 广播 (Broadcasting)
#             # 自动将 RGB 的 1x1 全局特征复制到 Pose 特征的空间尺寸 (H, W) 和时间尺寸 (T)
#             # x_pose.shape 通常是 (B, C, T, H, W)
#             x_rgb_lateral = x_rgb_lateral.expand(-1, -1, x_pose.shape[2], x_pose.shape[3], x_pose.shape[4])
#
#             x_pose = torch.cat((x_pose, x_rgb_lateral), dim=1)
#
#         # Layer 2
#         x_pose = self.pose_path.layer2(x_pose)
#         if hasattr(self.pose_path, 'layer2_lateral'):
#             feat = x_rgb.detach() if self.pose_detach else x_rgb
#             x_rgb_lateral = self.pose_path.layer2_lateral(feat)
#
#             if pose_drop_path:
#                 x_rgb_lateral = x_rgb_lateral.new_zeros(x_rgb_lateral.shape)
#
#             # [修改点 4] 广播
#             x_rgb_lateral = x_rgb_lateral.expand(-1, -1, x_pose.shape[2], x_pose.shape[3], x_pose.shape[4])
#
#             x_pose = torch.cat((x_pose, x_rgb_lateral), dim=1)
#
#         x_pose = self.pose_path.layer3(x_pose)
#
#         return (x_rgb, x_pose)
#
#     def train(self, mode=True):
#         super().train(mode)
#         self.training = True
#
#     def eval(self):
#         super().eval()
#         self.training = False
#
#
# # =========================================================
# # 升级版本 (Upgrade Version) - 支持冻结骨干，提分神器
# # =========================================================
#
# @BACKBONES.register_module()
# class ConvAcTransformer1(nn.Module):
#     """
#     版本 1.0: 支持冻结特征提取器 (Frozen Backbone)
#     """
#
#     def __init__(self,
#                  attention_heads: int,
#                  num_layers: int,
#                  num_classes: int,
#                  num_frames: int,
#                  drop_p: float,
#                  feature_extractor_name: str,
#                  learnable_pe: bool = False,
#                  frozen_backbone: bool = True):  # [新增] 默认开启冻结
#         super(ConvAcTransformer1, self).__init__()
#         self.feature_extractor_name = feature_extractor_name
#         self.attention_heads = attention_heads
#         self.num_classes = num_classes
#         self.num_layers = num_layers
#         self.num_frames = num_frames
#         self.learnable_pe = learnable_pe
#         self.drop_p = drop_p
#         self.frozen_backbone = frozen_backbone  # 记录冻结状态
#
#         self.feature_extract = FeatureExtractor(self.feature_extractor_name, model_weights='DEFAULT')
#
#         # [关键逻辑] 冻结特征提取器
#         if self.frozen_backbone:
#             for param in self.feature_extract.parameters():
#                 param.requires_grad = False
#             self.feature_extract.eval()  # 初始状态设为 eval
#
#         self.patch_embed = PatchEmbedding(self.feature_extract.d_model, learnable=self.learnable_pe,
#                                           max_len=self.num_frames, dropout=self.drop_p)
#
#         self.transformer_encoder = Encoder(seq_length=self.num_frames + 1,
#                                            num_layers=self.num_layers,
#                                            num_heads=self.attention_heads,
#                                            hidden_dim=self.feature_extract.d_model,
#                                            mlp_dim=self.feature_extract.d_model,
#                                            dropout=self.drop_p, attention_dropout=self.drop_p)
#
#         self.dropout = nn.Dropout(self.drop_p)
#         self.classification_head = nn.Linear(self.feature_extract.d_model, self.num_classes)
#
#     def forward(self, x):
#         # 如果冻结，这里 feature_extract 只做推理，不计算梯度
#         x = self.feature_extract(x)
#         x = self.patch_embed(x)
#         x = self.transformer_encoder(x)
#         cls_token = x[:, 0, :]
#         # 升维，兼容 I3DHead
#         output = cls_token.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         return output
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 kaiming_init(m)
#             elif isinstance(m, _BatchNorm):
#                 constant_init(m, 1)
#         # 冻结的骨干使用预训练权重，无需额外初始化
#
#     def train(self, mode=True):
#         """重写 train 方法，确保冻结层永远处于 eval 模式"""
#         super().train(mode)
#         self.training = True
#         if self.frozen_backbone:
#             # 这一步至关重要：锁死 BatchNorm
#             self.feature_extract.eval()
#
#
# @BACKBONES.register_module()
# class RGBPoseConv3DTransformer1(nn.Module):
#     """
#     版本 1.0: 使用 ConvAcTransformer1 (支持冻结)
#     """
#
#     def __init__(self,
#                  pretrained=None,
#                  speed_ratio=4,
#                  channel_ratio=4,
#                  rgb_detach=False,
#                  pose_detach=False,
#                  rgb_drop_path=0,
#                  pose_drop_path=0,
#                  rgb_pathway=dict(
#                      attention_heads=4,
#                      num_layers=4,
#                      num_classes=101,
#                      num_frames=50,
#                      drop_p=0.1,
#                      feature_extractor_name='wide_resnet50_2',
#                      learnable_pe=False,
#                      frozen_backbone=True  # 传递给 ConvAcTransformer1
#                  ),
#                  pose_pathway=dict(...)):  # 这里省略默认值，实际运行时传入即可
#
#         super().__init__()
#         self.pretrained = pretrained
#         self.speed_ratio = speed_ratio
#         self.channel_ratio = channel_ratio
#
#         # [修改] 使用新的 Transformer1 类
#         self.rgb_path = ConvAcTransformer1(**rgb_pathway)
#
#         self.pose_path = ResNet3dPathway(**pose_pathway)
#         self.rgb_detach = rgb_detach
#         self.pose_detach = pose_detach
#         assert 0 <= rgb_drop_path <= 1
#         assert 0 <= pose_drop_path <= 1
#         self.rgb_drop_path = rgb_drop_path
#         self.pose_drop_path = pose_drop_path
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 kaiming_init(m)
#             elif isinstance(m, _BatchNorm):
#                 constant_init(m, 1)
#         if isinstance(self.pretrained, str):
#             logger = get_root_logger()
#             print_log(f'load model from: {self.pretrained}', logger=logger)
#             load_checkpoint(self, self.pretrained, strict=True, logger=logger)
#         elif self.pretrained is None:
#             self.pose_path.init_weights()
#             self.rgb_path.init_weights()
#         else:
#             raise TypeError('pretrained must be a str or None')
#
#     def forward(self, imgs, heatmap_imgs):
#         if self.training:
#             rgb_drop_path = torch.rand(1) < self.rgb_drop_path
#             pose_drop_path = torch.rand(1) < self.pose_drop_path
#         else:
#             rgb_drop_path, pose_drop_path = False, False
#
#         # 1. RGB 分支 (5D Tensor)
#         x_rgb = self.rgb_path(imgs)
#
#         # 2. Pose 分支
#         x_pose = self.pose_path.conv1(heatmap_imgs)
#         x_pose = self.pose_path.maxpool(x_pose)
#
#         # Layer 1
#         x_pose = self.pose_path.layer1(x_pose)
#         if hasattr(self.pose_path, 'layer1_lateral'):
#             feat = x_rgb.detach() if self.pose_detach else x_rgb
#             x_rgb_lateral = self.pose_path.layer1_lateral(feat)
#             if pose_drop_path:
#                 x_rgb_lateral = x_rgb_lateral.new_zeros(x_rgb_lateral.shape)
#
#             # 广播 (Broadcast)
#             x_rgb_lateral = x_rgb_lateral.expand(-1, -1, x_pose.shape[2], x_pose.shape[3], x_pose.shape[4])
#             x_pose = torch.cat((x_pose, x_rgb_lateral), dim=1)
#
#         # Layer 2
#         x_pose = self.pose_path.layer2(x_pose)
#         if hasattr(self.pose_path, 'layer2_lateral'):
#             feat = x_rgb.detach() if self.pose_detach else x_rgb
#             x_rgb_lateral = self.pose_path.layer2_lateral(feat)
#             if pose_drop_path:
#                 x_rgb_lateral = x_rgb_lateral.new_zeros(x_rgb_lateral.shape)
#
#             # 广播 (Broadcast)
#             x_rgb_lateral = x_rgb_lateral.expand(-1, -1, x_pose.shape[2], x_pose.shape[3], x_pose.shape[4])
#             x_pose = torch.cat((x_pose, x_rgb_lateral), dim=1)
#
#         x_pose = self.pose_path.layer3(x_pose)
#         return (x_rgb, x_pose)
#
#     def train(self, mode=True):
#         super().train(mode)
#         self.training = True
#         # 确保子模块的 train 逻辑被调用（ConvAcTransformer1 会自己处理 freeze）
#         self.rgb_path.train(mode)
#         self.pose_path.train(mode)
#
#     def eval(self):
#         super().eval()
#         self.training = False


import torch
import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from mmcv.utils import _BatchNorm, print_log

from ...utils import get_root_logger
from ..builder import BACKBONES
from .resnet3d_slowfast import ResNet3dPathway

from torchvision import models
import math
from torchvision.models.vision_transformer import Encoder


# =========================================================
# 基础组件 (Shared Components)
# =========================================================

class PatchEmbedding(nn.Module):
    def __init__(self, d_model: int, learnable=False, dropout: float = 0.1, max_len: int = 15):
        super(PatchEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        frames = max_len + 1
        if learnable:
            self.pe = nn.Parameter(torch.randn((1, frames, d_model)))
        else:
            position = torch.arange(frames).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(1, frames, d_model)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)

    def forward(self, x):
        b, _, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class FeatureExtractor(nn.Module):
    def __init__(self, model_name: str, model_weights: str = 'DEFAULT'):
        super(FeatureExtractor, self).__init__()
        try:
            self.model = getattr(models, model_name)(weights=model_weights)
        except TypeError:
            self.model = getattr(models, model_name)(pretrained=(model_weights != None and model_weights != 'DEFAULT'))

        self.d_model = None

        if model_name in ['efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l', 'convnext_base',
                          'convnext_small', 'convnext_large']:
            self.d_model = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Identity()
        else:
            self.d_model = self.model.fc.in_features
            self.model.fc = nn.Identity()

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        b, f, _, _, _ = x.shape
        x = x.reshape(b * f, *x.size()[2:])
        x = self.model(x)
        x = x.reshape(b, f, *x.size()[1:])
        return x


# =========================================================
# 原始版本 (ConvAcTransformer) - 对应你当前的 Config
# =========================================================

@BACKBONES.register_module()
class ConvAcTransformer(nn.Module):
    def __init__(self,
                 attention_heads: int,
                 num_layers: int,
                 num_classes: int,
                 num_frames: int,
                 drop_p: float,
                 feature_extractor_name: str,
                 learnable_pe: bool = False):
        super(ConvAcTransformer, self).__init__()
        self.feature_extractor_name = feature_extractor_name
        self.attention_heads = attention_heads
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_frames = num_frames
        self.learnable_pe = learnable_pe
        self.drop_p = drop_p

        self.feature_extract = FeatureExtractor(self.feature_extractor_name, model_weights='DEFAULT')
        self.patch_embed = PatchEmbedding(self.feature_extract.d_model, learnable=self.learnable_pe,
                                          max_len=self.num_frames, dropout=self.drop_p)

        self.transformer_encoder = Encoder(seq_length=self.num_frames + 1,
                                           num_layers=self.num_layers,
                                           num_heads=self.attention_heads,
                                           hidden_dim=self.feature_extract.d_model,
                                           mlp_dim=self.feature_extract.d_model,
                                           dropout=self.drop_p, attention_dropout=self.drop_p)

        self.dropout = nn.Dropout(self.drop_p)
        self.classification_head = nn.Linear(self.feature_extract.d_model, self.num_classes)

    def forward(self, x):
        x = self.feature_extract(x)
        x = self.patch_embed(x)
        x = self.transformer_encoder(x)
        # 获取 CLS Token
        cls_token = x[:, 0, :]
        # 升维成 5D 张量 (B, C, 1, 1, 1) 以兼容 I3DHead
        output = cls_token.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return output

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, _BatchNorm):
                constant_init(m, 1)


@BACKBONES.register_module()
class RGBPoseConv3DTransformer(nn.Module):
    """Slowfast backbone with replaced RGB pathway as ConvAcTransformer."""

    def __init__(self,
                 pretrained=None,
                 speed_ratio=4,
                 channel_ratio=4,
                 rgb_detach=False,
                 pose_detach=False,
                 rgb_drop_path=0,
                 pose_drop_path=0,
                 rgb_pathway=dict(
                     attention_heads=4,
                     num_layers=4,
                     num_classes=101,
                     num_frames=50,
                     drop_p=0.1,
                     feature_extractor_name='wide_resnet50_2',
                     learnable_pe=False
                 ),
                 pose_pathway=dict(
                     num_stages=3,
                     stage_blocks=(4, 6, 3),
                     lateral=True,
                     lateral_inv=True,
                     lateral_infl=16,
                     lateral_activate=(0, 1, 1),
                     in_channels=17,
                     base_channels=32,
                     out_indices=(2,),
                     conv1_kernel=(1, 7, 7),
                     conv1_stride=(1, 1),
                     pool1_stride=(1, 1),
                     inflate=(0, 1, 1),
                     spatial_strides=(2, 2, 2),
                     temporal_strides=(1, 1, 1))):

        super().__init__()
        self.pretrained = pretrained
        self.speed_ratio = speed_ratio
        self.channel_ratio = channel_ratio

        # ConvAcTransformer replaces ResNet3dPathway for RGB
        self.rgb_path = ConvAcTransformer(**rgb_pathway)
        # Pose pathway remains unchanged
        self.pose_path = ResNet3dPathway(**pose_pathway)
        self.rgb_detach = rgb_detach
        self.pose_detach = pose_detach
        assert 0 <= rgb_drop_path <= 1
        assert 0 <= pose_drop_path <= 1
        self.rgb_drop_path = rgb_drop_path
        self.pose_drop_path = pose_drop_path

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, _BatchNorm):
                constant_init(m, 1)

        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            print_log(f'load model from: {self.pretrained}', logger=logger)
            load_checkpoint(self, self.pretrained, strict=True, logger=logger)
        elif self.pretrained is None:
            self.pose_path.init_weights()
            self.rgb_path.init_weights()
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, imgs, heatmap_imgs):
        if self.training:
            rgb_drop_path = torch.rand(1) < self.rgb_drop_path
            pose_drop_path = torch.rand(1) < self.pose_drop_path
        else:
            rgb_drop_path, pose_drop_path = False, False

        x_rgb = self.rgb_path(imgs)
        x_pose = self.pose_path.conv1(heatmap_imgs)
        x_pose = self.pose_path.maxpool(x_pose)
        x_pose = self.pose_path.layer1(x_pose)

        if hasattr(self.pose_path, 'layer1_lateral'):
            feat = x_rgb.detach() if self.pose_detach else x_rgb
            x_rgb_lateral = self.pose_path.layer1_lateral(feat)
            if pose_drop_path:
                x_rgb_lateral = x_rgb_lateral.new_zeros(x_rgb_lateral.shape)
            # 广播
            x_rgb_lateral = x_rgb_lateral.expand(-1, -1, x_pose.shape[2], x_pose.shape[3], x_pose.shape[4])
            x_pose = torch.cat((x_pose, x_rgb_lateral), dim=1)

        x_pose = self.pose_path.layer2(x_pose)
        if hasattr(self.pose_path, 'layer2_lateral'):
            feat = x_rgb.detach() if self.pose_detach else x_rgb
            x_rgb_lateral = self.pose_path.layer2_lateral(feat)
            if pose_drop_path:
                x_rgb_lateral = x_rgb_lateral.new_zeros(x_rgb_lateral.shape)
            # 广播
            x_rgb_lateral = x_rgb_lateral.expand(-1, -1, x_pose.shape[2], x_pose.shape[3], x_pose.shape[4])
            x_pose = torch.cat((x_pose, x_rgb_lateral), dim=1)

        x_pose = self.pose_path.layer3(x_pose)
        return (x_rgb, x_pose)

    def train(self, mode=True):
        super().train(mode)
        self.training = True

    def eval(self):
        super().eval()
        self.training = False


# =========================================================
# 升级版本 1 (修复了省略号，支持冻结参数)
# =========================================================

@BACKBONES.register_module()
class ConvAcTransformer1(nn.Module):
    """
    版本 1.0: 支持冻结特征提取器 (Frozen Backbone)
    """

    def __init__(self,
                 attention_heads: int,
                 num_layers: int,
                 num_classes: int,
                 num_frames: int,
                 drop_p: float,
                 feature_extractor_name: str,
                 learnable_pe: bool = False,
                 frozen_backbone: bool = False):  # [默认设为 False] 方便你先跑不冻结的
        super(ConvAcTransformer1, self).__init__()
        self.feature_extractor_name = feature_extractor_name
        self.attention_heads = attention_heads
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_frames = num_frames
        self.learnable_pe = learnable_pe
        self.drop_p = drop_p
        self.frozen_backbone = frozen_backbone

        self.feature_extract = FeatureExtractor(self.feature_extractor_name, model_weights='DEFAULT')

        # [逻辑] 冻结特征提取器
        if self.frozen_backbone:
            for param in self.feature_extract.parameters():
                param.requires_grad = False
            self.feature_extract.eval()

        self.patch_embed = PatchEmbedding(self.feature_extract.d_model, learnable=self.learnable_pe,
                                          max_len=self.num_frames, dropout=self.drop_p)

        self.transformer_encoder = Encoder(seq_length=self.num_frames + 1,
                                           num_layers=self.num_layers,
                                           num_heads=self.attention_heads,
                                           hidden_dim=self.feature_extract.d_model,
                                           mlp_dim=self.feature_extract.d_model,
                                           dropout=self.drop_p, attention_dropout=self.drop_p)

        self.dropout = nn.Dropout(self.drop_p)
        self.classification_head = nn.Linear(self.feature_extract.d_model, self.num_classes)

    def forward(self, x):
        x = self.feature_extract(x)
        x = self.patch_embed(x)
        x = self.transformer_encoder(x)
        cls_token = x[:, 0, :]
        # 升维
        output = cls_token.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return output

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, _BatchNorm):
                constant_init(m, 1)

    def train(self, mode=True):
        super().train(mode)
        self.training = True
        if self.frozen_backbone:
            self.feature_extract.eval()


@BACKBONES.register_module()
class RGBPoseConv3DTransformer1(nn.Module):
    """
    版本 1.0: 使用 ConvAcTransformer1
    """

    def __init__(self,
                 pretrained=None,
                 speed_ratio=4,
                 channel_ratio=4,
                 rgb_detach=False,
                 pose_detach=False,
                 rgb_drop_path=0,
                 pose_drop_path=0,
                 rgb_pathway=dict(
                     attention_heads=4,
                     num_layers=4,
                     num_classes=101,
                     num_frames=50,
                     drop_p=0.1,
                     feature_extractor_name='wide_resnet50_2',
                     learnable_pe=False,
                     frozen_backbone=False  # [默认 False]
                 ),
                 # [修正] 这里填充了完整的默认参数，消除了省略号报错
                 pose_pathway=dict(
                     num_stages=3,
                     stage_blocks=(4, 6, 3),
                     lateral=True,
                     lateral_inv=True,
                     lateral_infl=16,
                     lateral_activate=(0, 1, 1),
                     in_channels=17,
                     base_channels=32,
                     out_indices=(2,),
                     conv1_kernel=(1, 7, 7),
                     conv1_stride=(1, 1),
                     pool1_stride=(1, 1),
                     inflate=(0, 1, 1),
                     spatial_strides=(2, 2, 2),
                     temporal_strides=(1, 1, 1))):

        super().__init__()
        self.pretrained = pretrained
        self.speed_ratio = speed_ratio
        self.channel_ratio = channel_ratio

        # 使用新的 Transformer1 类
        self.rgb_path = ConvAcTransformer1(**rgb_pathway)

        self.pose_path = ResNet3dPathway(**pose_pathway)
        self.rgb_detach = rgb_detach
        self.pose_detach = pose_detach
        assert 0 <= rgb_drop_path <= 1
        assert 0 <= pose_drop_path <= 1
        self.rgb_drop_path = rgb_drop_path
        self.pose_drop_path = pose_drop_path

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, _BatchNorm):
                constant_init(m, 1)
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            print_log(f'load model from: {self.pretrained}', logger=logger)
            load_checkpoint(self, self.pretrained, strict=True, logger=logger)
        elif self.pretrained is None:
            self.pose_path.init_weights()
            self.rgb_path.init_weights()
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, imgs, heatmap_imgs):
        if self.training:
            rgb_drop_path = torch.rand(1) < self.rgb_drop_path
            pose_drop_path = torch.rand(1) < self.pose_drop_path
        else:
            rgb_drop_path, pose_drop_path = False, False

        # 1. RGB 分支
        x_rgb = self.rgb_path(imgs)

        # 2. Pose 分支
        x_pose = self.pose_path.conv1(heatmap_imgs)
        x_pose = self.pose_path.maxpool(x_pose)

        # Layer 1
        x_pose = self.pose_path.layer1(x_pose)
        if hasattr(self.pose_path, 'layer1_lateral'):
            feat = x_rgb.detach() if self.pose_detach else x_rgb
            x_rgb_lateral = self.pose_path.layer1_lateral(feat)
            if pose_drop_path:
                x_rgb_lateral = x_rgb_lateral.new_zeros(x_rgb_lateral.shape)
            x_rgb_lateral = x_rgb_lateral.expand(-1, -1, x_pose.shape[2], x_pose.shape[3], x_pose.shape[4])
            x_pose = torch.cat((x_pose, x_rgb_lateral), dim=1)

        # Layer 2
        x_pose = self.pose_path.layer2(x_pose)
        if hasattr(self.pose_path, 'layer2_lateral'):
            feat = x_rgb.detach() if self.pose_detach else x_rgb
            x_rgb_lateral = self.pose_path.layer2_lateral(feat)
            if pose_drop_path:
                x_rgb_lateral = x_rgb_lateral.new_zeros(x_rgb_lateral.shape)
            x_rgb_lateral = x_rgb_lateral.expand(-1, -1, x_pose.shape[2], x_pose.shape[3], x_pose.shape[4])
            x_pose = torch.cat((x_pose, x_rgb_lateral), dim=1)

        x_pose = self.pose_path.layer3(x_pose)
        return (x_rgb, x_pose)

    def train(self, mode=True):
        super().train(mode)
        self.training = True
        self.rgb_path.train(mode)
        self.pose_path.train(mode)

    def eval(self):
        super().eval()
        self.training = False