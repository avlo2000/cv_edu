import copy

import torch
from torch import nn
from torch.utils import data
from tqdm import tqdm


def lerp_model(model_a: nn.Sequential, model_b: nn.Sequential, weight):
    trg_model = copy.deepcopy(model_a)
    state_dict = trg_model.state_dict()
    for key, state in model_b.state_dict().items():
        state_dict[key] = (1.0 - weight) * state_dict[key] + weight * state
    trg_model.load_state_dict(state_dict)
    return trg_model


def objective_loss(model, loss_fn, data_loader, device):
    total_loss = 0.0
    for x, y in data_loader:
        x = x.to(device)
        y_pred = model(x)
        num_classes = y_pred.shape[1]
        y = nn.functional.one_hot(y, num_classes=num_classes).to(device, dtype=torch.float32)
        loss = loss_fn(y_pred, y)
        total_loss += loss.item()
    return total_loss


def loss_drill(
        model_a: nn.Sequential,
        model_b: nn.Sequential,
        loss_fn: nn.Module,
        data_loader: data.DataLoader,
        ticks_count: int,
        device):
    weights = torch.linspace(0.0, 1.0, ticks_count)
    losses = torch.empty_like(weights)
    for i, w in tqdm(enumerate(weights), desc='Eval on A B lerps', total=weights.shape.numel()):
        model = lerp_model(model_a, model_b, w)
        losses[i] = objective_loss(model, loss_fn, data_loader, device)
    return weights, losses
