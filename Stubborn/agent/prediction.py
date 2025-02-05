import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import mmseg

import mmcv
import os.path as osp
import numpy as np

from scipy.special import expit
from mmcv import Config

from mmseg.datasets.builder import PIPELINES
from mmseg.models.builder import LOSSES
from mmseg.models.losses.utils import weighted_loss
from mmseg.apis import set_random_seed, init_segmentor
from mmseg.utils import get_device
from mmseg.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter


def sigmoid(x):
    return expit(x)
    
    
@PIPELINES.register_module()
class LoadMap(object):
    """Load semantic maps from numpy array.
    Required keys are "full_map". Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).
    """

    def __init__(self,
                 to_float32=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='np'):
        self.to_float32 = to_float32
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.
        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.
        Returns:
            dict: The dict contains loaded image and meta information.
        """

        img = results['full_map'].transpose(1, 2, 0)
        img = img.astype(np.float32)

        results['filename'] = None
        results['ori_filename'] = None
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = img.shape[0]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str
    
    
@weighted_loss
def my_loss(pred, target):
    target = torch.permute(target, (0, 3, 1, 2))
    assert pred.size() == target.size() and target.numel() > 0
    wts = [36.64341412, 30.19407855, 106.23704066, 25.58503269, 100.4556983, 167.64383946]
    pos_weight = torch.ones(6, 720, 720).to(pred.device) #torch.ones(6)
    for i, wt in enumerate(wts):
        pos_weight[i] = wts[i]
    loss = F.binary_cross_entropy_with_logits(pred, target / 255., reduction='none', pos_weight=pos_weight)
    return loss

@LOSSES.register_module
class MyLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(MyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * my_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss
    
    @property
    def loss_name(self):
        return 'loss_bce'

    
def inference_smp(model, full_map):
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
    # prepare data
    test_pipeline = [LoadMap()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    data = dict(full_map=full_map)
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


class SemOccPred():
    def __init__(self, args):
        self.args = args
        ckpt = osp.join('./Stubborn/smp_wts_temp.pth')
    
        cfg = Config.fromfile('./Stubborn/agent/utils/smp_cfg_temp.py')
        if args.num_sem_categories == 23:
            ckpt = osp.join('./Stubborn/smprn_wts_temp.pth')
    
            cfg = Config.fromfile('./Stubborn/agent/utils/smprn_cfg_temp.py')

        # build the model from a config file and a checkpoint file
        self.model = init_segmentor(cfg, checkpoint=ckpt, device=('cuda:'+ str(args.sem_gpu_id)) if args else 'cuda:0')
        self.model.eval()

        self.model.cfg = cfg

        
    def get_prediction(self, full_map):
        
        result = inference_smp(self.model, full_map)
        return sigmoid(result[0])
    
    
    
