from typing import Callable, Sequence

import torch
from torch import nn
import torch.nn.functional as fn


class DNN(nn.Module):
    def __init__(self,
                 layer_dims: Sequence[int],
                 activation_module: nn.Module = nn.ReLU
                 ):
        super().__init__()
        self._model = nn.Sequential()
        for dim0, dim1 in zip(layer_dims, layer_dims[1:]):
            self._model.append(nn.Linear(dim0, dim1))
            self._model.append(activation_module)

    def forward(self, state: torch.Tensor):
        return self._model(state)


class DQN:
    def __init__(self, policy_approx: DNN):
        pass

    def act(self, state: torch.Tensor):
        pass
