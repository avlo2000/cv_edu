from typing import List

import torch
import torch.nn as nn


class Yolo5Backbone(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        yolo5nano = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True).to('cpu')
        try:
            backbone_layers = yolo5nano.model.model.model[:9]
        except AttributeError:
            backbone_layers = yolo5nano.model[:9]
        self.__backbone_pre = backbone_layers[0]
        self.__backbone_pyr = nn.ModuleList()
        for l0, l1 in zip(backbone_layers[1::2], backbone_layers[2::2]):
            l0.act.inplace = False
            self.__backbone_pyr.append(nn.Sequential(l0, l1))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.__backbone_pre(x)
        pyramid = []
        for layer_pyr in self.__backbone_pyr:
            x = layer_pyr(x)
            pyramid.append(x)
        return list(reversed(pyramid))


if __name__ == '__main__':
    model = Yolo5Backbone()
    w, h = 1280, 720
    dummy_inp = torch.zeros([1, 3, w, h]).to('cuda')

    y = model(dummy_inp)

    for k, v in model.named_parameters():
        print(k)

