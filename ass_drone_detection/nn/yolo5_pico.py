from collections import OrderedDict
from typing import List, Sequence, Callable

import torch
from torch import nn
import torch.nn.functional as fn
from torchvision.ops import FeaturePyramidNetwork, boxes

from yolo5_backbone import Yolo5Backbone


class DetectionHead(nn.Module):
    def __init__(self, w: int, h: int, in_channels: int, nx: int, ny: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv_post = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(64, 5, kernel_size=1),
        )
        self.nx, self.ny = nx, ny

        wh = torch.tensor([w, h])
        pwh = wh / torch.tensor([nx, ny])
        self.register_buffer('_pwh', pwh)
        self.register_buffer('_wh', wh)
        grid = self._create_grid(nx, ny)
        self.register_buffer('_grid', grid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_post(x)
        x[:, :4] = fn.sigmoid(x[:, :4])
        conf = x[:, 4]
        conf_flat = fn.softmax(conf.flatten(1, 2))
        conf = torch.unflatten(conf_flat, dim=1, sizes=(x.size(2), x.size(3)))

        xy = (x[:, 0:2] - 0.5) * self._pwh.view(1, 2, 1, 1) + self._grid[None, :]
        wh = ((2 * x[:, 2:4]) ** 2) * self._pwh.view(1, 2, 1, 1)
        return torch.cat((xy, wh, conf[:, None]), dim=1)

    def expected_scores(self, bbox: torch.Tensor):
        grid_bbox = torch.cat([self._grid, self._grid + self._pwh.view(2, 1, 1)], dim=0)
        flat_grid_bbox = torch.flatten(grid_bbox, 1, 2).permute(1, 0)
        bbox[:, 2] += bbox[:, 0]
        bbox[:, 3] += bbox[:, 1]
        iou = boxes.box_iou(bbox, flat_grid_bbox)
        iou_unflatten = torch.unflatten(iou, 1, (grid_bbox.size(1), grid_bbox.size(2)))
        mask = torch.zeros_like(iou_unflatten)
        for i in range(iou_unflatten.size(0)):
            mx = iou_unflatten[i].max()
            mask[i, mx == iou_unflatten[i]] = 1.0
        return iou_unflatten

    def _create_grid(self, nx, ny):
        x_lin = torch.linspace(0, self._wh[0] - self._pwh[0], nx)
        y_lin = torch.linspace(0, self._wh[1] - self._pwh[1], ny)
        xx, yy = torch.meshgrid(x_lin, y_lin)
        return torch.stack([xx, yy])


class Yolo5Pico(nn.Module):
    def __init__(self, w: int, h: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.w, self.h = w, h
        self.backbone = Yolo5Backbone()
        dummy_inp = torch.zeros([1, 3, w, h])
        dummy_feats = self.backbone(dummy_inp)[:-1]
        self.fpn = FeaturePyramidNetwork(
            [x.size(1) for x in dummy_feats],
            out_channels=64,
        )
        dummy_feats_dict = self.__feats_list_to_dict(dummy_feats)
        dummy_fpn_out = self.fpn(dummy_feats_dict).values()
        self.detection_heads = nn.ModuleList([DetectionHead(w, h, 64, y.shape[2], y.shape[3]) for y in dummy_fpn_out])
        heads = [head(x) for head, x in zip(self.detection_heads, dummy_fpn_out)]
        numel_per_head = [head.size(2) * head.size(3) for head in heads]
        self.pred_head_indexes = numel_per_head

    def build_residual_loss(self) -> Callable[[List[torch.Tensor], torch.Tensor], torch.Tensor]:
        score_w = 100
        xy_w = 5
        wh_w = 5

        def residual(pred: List[torch.Tensor], expected_boxes: torch.Tensor) -> torch.Tensor:
            """
            :param pred: raw Yolo prediction Bx5xN
            :param expected_boxes: Bx4 bboxes in format 'xywh'
            :returns: calculated residual loss
            """
            res = torch.tensor(0.0, device=expected_boxes.device)
            for i, head_pred in enumerate(pred):
                expected_scores = self.detection_heads[i].expected_scores(expected_boxes.clone())
                scores = head_pred[:, 4]
                bboxes = head_pred[:, :4]
                expected_head_bb = expected_boxes[..., None, None].repeat(1, 1, *expected_scores.shape[1:])
                xy_loss = fn.l1_loss(bboxes[:, :2, :, :], expected_head_bb[:, :2, :, :], reduction='none') * expected_scores[:, None]
                wh_loss = fn.l1_loss(bboxes[:, 2:, :, :], expected_head_bb[:, 2:, :, :], reduction='none') * expected_scores[:, None]
                res += (score_w * fn.binary_cross_entropy(scores, expected_scores, reduction='mean') +
                        xy_w * xy_loss.mean() +
                        wh_w * wh_loss.mean())
            return res
        return residual

    def forward(self, img: torch.Tensor) -> List[torch.Tensor]:
        pyramid = self.backbone(img)[:-1]
        fpn_out = self.fpn(self.__feats_list_to_dict(pyramid)).values()
        bboxes = [head(x) for head, x in zip(self.detection_heads, fpn_out)]
        return bboxes

    def __feats_list_to_dict(self, feats):
        dict_feats = OrderedDict()
        for i, f in enumerate(feats):
            dict_feats[f'feat{i}'] = f
        return dict_feats
