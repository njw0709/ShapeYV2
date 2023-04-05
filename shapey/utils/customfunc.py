import torch
import torch.nn as nn


def pearsonr(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val


def pearsonr_batch(x_batch, y_batch, prenormalized=False):
    if not prenormalized:
        mean_x_b = torch.mean(x_batch, 1, True)
        mean_y_b = torch.mean(y_batch, 1, True)
        xm_b = x_batch.sub(mean_x_b)
        ym_b = y_batch.sub(mean_y_b)
        r_num_b = (xm_b * ym_b).sum(1, keepdim=True)
        r_den_b = torch.norm(xm_b, dim=1, keepdim=True) * torch.norm(
            ym_b, dim=1, keepdim=True
        )
        r_vals = r_num_b / r_den_b
    else:
        r_vals = (x_batch * y_batch).sum(1, keepdim=True)
    return r_vals


def ln_batch(x_batch, y_batch, n=1):
    lndist = nn.PairwiseDistance(p=n, keepdim=True)
    d = lndist(x_batch, y_batch)
    return d


def ln_cdist(x_batch, y_batch, n=1):
    d_mat = torch.cdist(x_batch, y_batch, p=n)
    return d_mat
