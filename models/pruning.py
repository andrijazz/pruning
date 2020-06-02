import numpy as np
import torch
import torch.nn as nn


def matrix_topk(x, k):
    H, W = x.shape

    x = x.view(-1)
    _, indices = x.topk(k)
    indices_cat = torch.cat(((indices // W).unsqueeze(1), (indices % W).unsqueeze(1)), dim=1)
    indices_cat = indices_cat.to('cpu').numpy()
    return indices_cat


def weight_pruning(model, k):
    zeroed_weights = 0
    if k == 0:
        return model, zeroed_weights

    for i in range(len(model.net) - 1):
        if not isinstance(model.net[i], nn.Linear):
            continue

        m = model.net[i].weight.shape[0] * model.net[i].weight.shape[1]
        n = (m // 100) * k
        W = -torch.abs(model.net[i].weight.data)
        indices = matrix_topk(W, n)
        model.net[i].weight.data[indices[:, 0], indices[:, 1]] = 0.
        zeroed_weights += indices.shape[0]
    return model, zeroed_weights


def unit_pruning(model, k):
    zeroed_weights = 0
    if k == 0:
        return model, zeroed_weights

    for i in range(len(model.net) - 1):
        if not isinstance(model.net[i], nn.Linear):
            continue

        r = model.net[i].weight.shape[0]
        c = model.net[i].weight.shape[1]
        n = (c // 100) * k
        W = torch.abs(model.net[i].weight.data)
        x = torch.sum(W * W, dim=0)
        _, indices = (-x).topk(n)
        model.net[i].weight.data[:, indices] = 0.
        zeroed_weights += r * n
    return model, zeroed_weights


def accuracy(predictions, gt):
    """

    @param predictions:
    @param gt:
    @return:
    """
    m = gt.shape[0]
    acc = np.sum(predictions == gt) / m
    return acc
