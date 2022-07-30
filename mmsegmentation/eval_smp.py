import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import mmseg

import mmcv
import os.path as osp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from mmcv import Config
from mmseg.apis import set_random_seed, init_segmentor
from mmseg.utils import get_device
from mmseg.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter

from train_smp import MyLoss, SemMapDataset, LoadMapFromFile
from constants import id_color, common_cls


def sigmoid(x):
    return 1/(1 + np.exp(-x))
    
    
def inference_smp(model, imgfile, t_idx):
    """Inference image(s) with the segmentor.
    Args:
        model (nn.Module): The loaded segmentor.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.
    Returns:
        (list[Tensor]): The segmentation result.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    img_info = dict(filename = imgfile, t_idx = t_idx)
    # prepare data
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = dict(img_info=img_info, seg_fields=[])
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]
    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


def visualize_obj_preds(pred, cls_ids, z_map, obj_map, mask):
    pred = sigmoid(pred[cls_ids])
    max_p = np.array([np.max(p) for p in pred])
    for i in range(len(cls_ids)):
        pred[i] /= max_p[i]
    
    z_map = z_map
    obj_map = obj_map
    mask = mask
    rgb = np.zeros((pred[0].shape[0], pred[0].shape[1], 3), dtype=float)
    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            rgb[i, j] = z_map[i, j]  * 0.8
            if np.sum(obj_map[:, i, j]) > 0:
                rgb[i, j] = id_color[np.argmax(obj_map[:, i, j])]/255 
            elif mask[i, j]:
                rgb[i, j] = max(rgb[i, j][0], 0.4)
    
    fig, axs = plt.subplots(1, len(cls_ids) + 1, figsize=(4* len(cls_ids), 4))
    axs[0].imshow(np.clip(rgb, 0, 1))
    axs[0].set_title('GT')
    axs[0].axis('off')
    for c in range(len(cls_ids)):
        pred_rgb = np.copy(rgb)
        pred_rgb[np.logical_not(mask.astype(bool))] = 0
        for i in range(rgb.shape[0]):
            for j in range(rgb.shape[1]):
                if mask[i, j] < 1:
                    pred_rgb[i, j] += pred[c, i, j] * id_color[cls_ids[c]]/255
                    if np.argmax(obj_map[:, i, j]) == cls_ids[c] and np.sum(obj_map[:, i, j]):
                        pred_rgb[i, j] = [1, 1, 0]
        axs[c + 1].imshow(np.clip(pred_rgb, 0, 1))
        axs[c + 1].axis('off')
        axs[c + 1].set_title('%s (%.4f max)' % (common_cls[cls_ids[c]], max_p[c]))
    plt.tight_layout()
    
    
def nearest_dist(pred, gt, voxel_size=0.05):
    """
    Takes prediction heatmap for a single class and ground-truth map for that class
    and returns distance from the prediction maximum to the nearest positive
    ground-truth pixel. Returns -1 if no object in ground-truth.
    """
    pred_loc = np.argmax(pred)
    pred_loc = pred_loc // pred.shape[1], pred_loc % pred.shape[1]
    
    gt_locs = np.where(gt > 0)
    if len(gt_locs[0]) == 0:
        return -1
    
    sqdist = (gt_locs[0] - pred_loc[0])**2 + (gt_locs[1] - pred_loc[1])**2
    nearest = np.min(sqdist)
    
    return np.sqrt(float(nearest)) * voxel_size


def neg_log_likelihood(pred, gt):
    """
    Takes prediction heatmap for a single class and ground-truth map for that class
    and returns the predicted negative log-likelihood averaged over ground-truth pixels.
    Lower is better. Returns -1 if no object in ground-truth.
    """
    # Converting to likelihood
    pred = sigmoid(pred)
    pred /= np.sum(pred)
    
    gt_locs = np.where(gt > 0)
    if len(gt_locs[0]) == 0:
        return -1
    
    return -np.mean(np.log(pred[gt_locs]))


if __name__ == '__main__':
    
    out_dir = './work_dirs/smp_r50'
    ckpt = osp.join(out_dir, 'iter_10000.pth')
    
    cfg = Config.fromfile(osp.join(out_dir, 'cfg.py'))
    # print(f'Config:\n{cfg.pretty_text}')
    
    # build the model from a config file and a checkpoint file
    model = init_segmentor(cfg, checkpoint=ckpt, device='cuda:0')
    # Add an attribute for visualization convenience
    model.CLASSES = common_cls

    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    model.eval()

    model.cfg = cfg

    # Save some qualitative results
    for i in [5]:
        for t_idx in range(4):
            print(i, t_idx)
            result = inference_smp(model, '../data/saved_maps/val/f%05d.npz' % i, t_idx=t_idx)
            pred = result[0]
            xd = np.load('../data/saved_maps/val/f%05d.npz' % i)['maps'] / 255.
            z_map = xd[-1, 0]
            obj_map = xd[-1, 4:]
            mask = xd[t_idx, 1] > 0
            visualize_obj_preds(pred, [0, 1, 2, 3, 4, 5], z_map, obj_map, mask)
            plt.savefig(osp.join(out_dir, 'qual%d_%d.png' % (i, t_idx)))
            plt.close()
            
#     # MSE and NLL evaluation 
#     dists = [[] for c in range(6)]
#     nlls = [[] for c in range(6)] 
#     for i in range(500):
#         if i % 50 == 0:
#             print(i)
#         mf =  np.load('../data/saved_maps/val/f%05d.npz' % i)['maps']/ 255.
#         z_map = mf[-1, 0]
#         obj_map = mf[-1, 4:]
#         for t_idx in range(4):
            
#             result = inference_smp(model, '../data/saved_maps/val/f%05d.npz' % i, t_idx=t_idx)
#             for c in range(6):
#                 pred = result[0][c]
#                 gt = obj_map[c]
#                 dist = nearest_dist(pred, gt)
#                 nll = neg_log_likelihood(pred, gt)
#                 if dist >= 0:
#                     dists[c].append(dist)
#                     nlls[c].append(nll)
                
#     for c in range(6):
#         print('%12s: %.3f MSE, %.3f NLL, %d occurrences' % (common_cls[c], np.mean(dists[c]), np.mean(nlls[c]), len(dists[c])))
#     all_dists = sum(dists, [])
#     all_nlls = sum(nlls, [])
#     print('-' * 60)
#     print('%12s: %.3f MSE, %.3f NLL, %d occurrences' % ('ALL CLASSES', np.mean(all_dists), np.mean(all_nlls),len(all_dists)))