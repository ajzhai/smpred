# From Vince Cartillier's Semantic MapNet

# Original Rednet
# https://github.com/JindongJiang/RedNet
# Vince's Port
# https://github.com/vincentcartillier/Semantic-MapNet

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

# The following code is largely borrowed from
# https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py and
# https://github.com/facebookresearch/detectron2/blob/master/demo/predictor.py

import argparse
import time

import torch
import numpy as np

from detectron2.config import get_cfg, LazyConfig, instantiate
from detectron2.engine.defaults import create_ddp_model
from detectron2.model_zoo import get_config
from detectron2.utils.logger import setup_logger
from detectron2.data.catalog import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import ColorMode, Visualizer
import detectron2.data.transforms as T
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation


from constants import hm3d_to_ade, coco_to_hm3d, coco_categories_mapping,fourty221, twentyone240,compatible_dict, white_list,black_list


# Resnet model urls
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def debug_tensor(label, tensor):
    print(label, tensor.size(), tensor.mean().item(), tensor.std().item())

class RedNet(nn.Module):
    def __init__(self, num_classes=40, pretrained=False):

        super().__init__()

        block = Bottleneck
        transblock = TransBasicBlock
        layers = [3, 4, 6, 3]
        # original resnet
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # resnet for depth channel
        self.inplanes = 64
        self.conv1_d = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                 bias=False)
        self.bn1_d = nn.BatchNorm2d(64)
        self.layer1_d = self._make_layer(block, 64, layers[0])
        self.layer2_d = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_d = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_d = self._make_layer(block, 512, layers[3], stride=2)

        self.inplanes = 512
        self.deconv1 = self._make_transpose(transblock, 256, 6, stride=2)
        self.deconv2 = self._make_transpose(transblock, 128, 4, stride=2)
        self.deconv3 = self._make_transpose(transblock, 64, 3, stride=2)
        self.deconv4 = self._make_transpose(transblock, 64, 3, stride=2)

        self.agant0 = self._make_agant_layer(64, 64)
        self.agant1 = self._make_agant_layer(64 * 4, 64)
        self.agant2 = self._make_agant_layer(128 * 4, 128)
        self.agant3 = self._make_agant_layer(256 * 4, 256)
        self.agant4 = self._make_agant_layer(512 * 4, 512)

        # final block
        self.inplanes = 64
        self.final_conv = self._make_transpose(transblock, 64, 3)

        self.final_deconv_custom = nn.ConvTranspose2d(self.inplanes, num_classes, kernel_size=2,
                                               stride=2, padding=0, bias=True)

        self.out5_conv_custom = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, bias=True)
        self.out4_conv_custom = nn.Conv2d(128, num_classes, kernel_size=1, stride=1, bias=True)
        self.out3_conv_custom = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, bias=True)
        self.out2_conv_custom = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, bias=True)

        if pretrained:
            self._load_resnet_pretrained()

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()



    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_transpose(self, block, planes, blocks, stride=1):

        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)

    def _make_agant_layer(self, inplanes, planes):

        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def _load_resnet_pretrained(self):
        pretrain_dict = model_zoo.load_url(model_urls['resnet50'])
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                if k.startswith('conv1'):  # the first conv_op
                    model_dict[k] = v
                    model_dict[k.replace('conv1', 'conv1_d')] = torch.mean(v, 1).data. \
                        view_as(state_dict[k.replace('conv1', 'conv1_d')])

                elif k.startswith('bn1'):
                    model_dict[k] = v
                    model_dict[k.replace('bn1', 'bn1_d')] = v
                elif k.startswith('layer'):
                    model_dict[k] = v
                    model_dict[k[:6] + '_d' + k[6:]] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def forward_downsample(self, rgb, depth):
        x = self.conv1(rgb)
        x = self.bn1(x)
        x = self.relu(x)
        depth = self.conv1_d(depth)
        depth = self.bn1_d(depth)
        depth = self.relu(depth)

        fuse0 = x + depth

        x = self.maxpool(fuse0)
        depth = self.maxpool(depth)

        # block 1
        x = self.layer1(x)
        # debug_tensor('post1', depth)
        # for i, mod in enumerate(self.layer1_d):
        #     depth = mod(depth)
        #     debug_tensor(f'post1d-{i}', depth)

        depth = self.layer1_d(depth)
        # debug_tensor('post1d', depth)

        fuse1 = x + depth
        # block 2
        x = self.layer2(fuse1)
        depth = self.layer2_d(depth)
        fuse2 = x + depth
        # block 3
        x = self.layer3(fuse2)
        depth = self.layer3_d(depth)
        fuse3 = x + depth
        # block 4
        x = self.layer4(fuse3)
        depth = self.layer4_d(depth)
        fuse4 = x + depth

        agant4 = self.agant4(fuse4)

        return fuse0, fuse1, fuse2, fuse3, agant4

    def forward_upsample(self, fuse0, fuse1, fuse2, fuse3, agant4):

        # upsample 1
        x = self.deconv1(agant4)
        if self.training:
            out5 = self.out5_conv_custom(x)
        x = x + self.agant3(fuse3)
        # upsample 2
        x = self.deconv2(x)
        if self.training:
            out4 = self.out4_conv_custom(x)
        x = x + self.agant2(fuse2)
        # upsample 3
        x = self.deconv3(x)
        if self.training:
            out3 = self.out3_conv_custom(x)
        x = x + self.agant1(fuse1)
        # upsample 4
        x = self.deconv4(x)
        if self.training:
            out2 = self.out2_conv_custom(x)
        x = x + self.agant0(fuse0)
        # final
        x = self.final_conv(x)

        last_layer = x

        out = self.final_deconv_custom(x)

        if self.training:
            return out, out2, out3, out4, out5

        return out, last_layer

    def forward(self, rgb, depth):

        fuses = self.forward_downsample(rgb, depth)

        # We only need predictions.
        # features_encoder = fuses[-1]
        # scores, features_lastlayer = self.forward_upsample(*fuses)
        scores, *_ = self.forward_upsample(*fuses)
        # debug_tensor('scores', scores)
        # return features_encoder, features_lastlayer, scores
        return scores


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class TransBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out

class BatchNormalize(nn.Module):
    r"""
        I can't believe this isn't supported
        https://github.com/pytorch/vision/issues/157
    """

    def __init__(self, mean, std, device):
        super().__init__()
        self.mean = torch.tensor(mean, device=device)[None, :, None, None]
        self.std = torch.tensor(std, device=device)[None, :, None, None]

    def forward(self, x):
        # mean = torch.tensor(self.mean, device=x.device)
        # std = torch.tensor(self.std, device=x.device)
        # mean = self.mean[None, :, None, None]
        # std = self.std[None, :, None, None]
        return (x - self.mean) / self.std
        # x.sub_(self.mean).div_(self.std)
        # return x

class RedNetResizeWrapper(nn.Module):
    def __init__(self, device, resize=True, stabilize=False):
        super().__init__()
        self.rednet = RedNet()
        self.semmap_rgb_norm = BatchNormalize(
            mean=[0.493, 0.468, 0.438],
            std=[0.544, 0.521, 0.499],
            device=device
        )
        self.semmap_depth_norm = BatchNormalize(
            mean=[0.213],
            std=[0.285],
            device=device
        )
        self.pretrained_size = (480, 640)
        self.resize = resize
        self.stabilize = stabilize

    def forward(self, rgb, depth):
        r"""
            Args:
                Raw sensor inputs.
                rgb: B x H=256 x W=256 x 3
                depth: B x H x W x 1
            Returns:
                semantic: drop-in replacement for default semantic sensor. B x H x W  (no channel, for some reason)
        """
        # Not quite sure what depth is produced here. Let's just check
        if self.resize:
            _, og_h, og_w, _ = rgb.size() # b h w c
        rgb = rgb.permute(0, 3, 1, 2)
        depth = depth.permute(0, 3, 1, 2)
        rgb = rgb.float() / 255
        if self.resize:
            rgb = F.interpolate(rgb, self.pretrained_size, mode='bilinear')
            depth = F.interpolate(depth, self.pretrained_size, mode='nearest')

        rgb = self.semmap_rgb_norm(rgb)

        depth_clip = (depth < 1.0).squeeze(1)
        # depth_clip = ((depth < 1.0) & (depth > 0.0)).squeeze(1)
        depth = self.semmap_depth_norm(depth)
        with torch.no_grad():
            scores = self.rednet(rgb, depth)
            pred = (torch.max(scores, 1)[1] ) # B x 480 x 640
            #print("pred shape",pred,pred.shape)
            return scores, pred[0]
        if self.stabilize: # downsample tiny
            # Mask out out of depth samples
            pred[~depth_clip] = -1 # 41 is UNK in MP3D, but hab-sim renders with -1
            # pred = F.interpolate(pred.unsqueeze(1), (15, 20), mode='nearest').squeeze(1)
        if self.resize:
            pred = F.interpolate(pred.unsqueeze(1), (og_h, og_w), mode='nearest')

        return pred.long().squeeze(1)

def load_rednet(device, ckpt="", resize=True, stabilize=False):
    if not os.path.isfile(ckpt):
        raise Exception(f"invalid path {ckpt} provided for rednet weights")

    model = RedNetResizeWrapper(device, resize=resize, stabilize=stabilize).to(device)

    print("=> loading RedNet checkpoint '{}'".format(ckpt))
    if device.type == 'cuda':
        checkpoint = torch.load(ckpt, map_location='cpu')
    else:
        checkpoint = torch.load(ckpt, map_location=lambda storage, loc: storage)

    state_dict = checkpoint['model_state']
    prefix = 'module.'
    state_dict = {
        (k[len(prefix):] if k[:len(prefix)] == prefix else k): v for k, v in state_dict.items()
    }
    model.rednet.load_state_dict(state_dict)
    print("=> loaded checkpoint '{}' (epoch {})"
            .format(ckpt, checkpoint['epoch']))

    return model


class SemanticPredRedNet():

    def __init__(self, args):
        self.segmentation_model = load_rednet(args.device,ckpt = args.checkpt, resize = True)
        self.segmentation_model.eval()
        self.args = args
        self.all_labels = set()
        self.threshold = args.sem_pred_prob_thr
        self.gt_mask = None
        self.goal_cat = None

    def get_prediction(self, img,depth,goal_cat=None):
        args = self.args
        #image_list = []
        img = img[np.newaxis, :, :, :]
        depth = depth[np.newaxis, :, :, :]
        #print("input shape is ",img.shape)
        #print(self.args.device)
        img = torch.from_numpy(img).float().to(self.args.device)
        depth = torch.from_numpy(depth).float().to(self.args.device)
        output, mask = self.segmentation_model(img,depth)
        #print("output shape is",output.shape)
        output = output[0]
        output = output *0.1
        output[output<self.threshold] = 0 #0.9: 30 1.1: 26
        semantic_input = np.zeros((img.shape[1], img.shape[2], 23 ))
        for i in range(0, 40):
            if i in fourty221.keys():
                output[i][mask != i] = 0
                j = fourty221[i]
                if (not (self.gt_mask is None)) and j == self.goal_cat:
                    semantic_input[:,:,j] += np.copy(self.gt_mask)
                    self.gt_mask = None
                else:
                    semantic_input[:, :, j] += (
                        output[i]).cpu().numpy()

        return semantic_input, mask

    def set_gt_mask(self,gt_mask,goal_cat): # goal cat is as it is in fourty221 mapping
        self.gt_mask = gt_mask
        self.goal_cat = goal_cat

class QuickSemanticPredRedNet():

    def __init__(self, args):
        self.segmentation_model = load_rednet(args.device,ckpt = args.checkpt, resize = True)
        self.segmentation_model.eval()
        self.args = args
        self.threshold = 0.7
        self.all_labels = set()
        self.gt_mask = None

    def get_conflict(self,output,goal_cat,ori_goal):
        output = torch.clone(output)
        output[goal_cat] *= 0
        for i in range(40):
            if i not in fourty221.keys():
                output[i] *= 0
            else:
                j = fourty221[i]
                if ori_goal in compatible_dict.keys() and j in compatible_dict[ori_goal]:
                    output[i] *= 0
        output,_ = torch.max(output,dim = 0)
        output *= 0.1
        output[output < 0.9] = 0
        return output.cpu().numpy()
    #only care about objects in blacklist
    def get_black_white_list(self, output,goal_cat,ori_goal,black_list):
        if ori_goal not in black_list.keys():
            return np.zeros(output[0].shape)
        siz = len(black_list[ori_goal])
        ans = torch.zeros((siz,output.shape[1],output.shape[2])).to(self.args.device)
        id = 0
        for i in black_list[ori_goal]:
            ans[id] = torch.clone( output[twentyone240[i]])
            id += 1


        ans, _ = torch.max(ans, dim=0)
        ans *= 0.1
        ans[ans < 0.9] = 0
        return ans.cpu().numpy()

    def get_prediction(self, img,depth,goal_cat):
        ori_goal = goal_cat
        goal_cat = twentyone240[goal_cat]
        args = self.args
        #image_list = []
        img = img[np.newaxis, :, :, :]
        depth = depth[np.newaxis, :, :, :]
        #print("input shape is ",img.shape)
        #print(self.args.device)
        img = torch.from_numpy(img).float().to(self.args.device)
        depth = torch.from_numpy(depth).float().to(self.args.device)
        output, mask = self.segmentation_model(img,depth)
        #print("output shape is",output.shape)
        output = output[0]
        output[goal_cat] *= 0.1
        max_score = torch.max(output[goal_cat]) - 0.05
        if self.threshold > max_score:
            max_score = self.threshold


        output[goal_cat][output[goal_cat] < max_score] = 0

        semantic_input = np.zeros((img.shape[1], img.shape[2], 5 + self.args.use_gt_mask ))
        semantic_input[:,:,0] = output[goal_cat].cpu().numpy()
        if self.gt_mask is not None:
            semantic_input[:,:,4] = self.gt_mask
        if self.args.record_conflict == 1:
            semantic_input[:,:,1] = self.get_conflict(output,goal_cat,ori_goal)
        semantic_input[:,:,2] = self.get_black_white_list(output,goal_cat,ori_goal,black_list)
        semantic_input[:,:,3] = self.get_black_white_list(output,goal_cat,ori_goal,white_list)
        return semantic_input

    def set_gt_mask(self,gt_mask):
        self.gt_mask = gt_mask


def compress_sem_map(sem_map):
    c_map = np.zeros((sem_map.shape[1], sem_map.shape[2]))
    for i in range(sem_map.shape[0]):
        c_map[sem_map[i] > 0.] = i + 1
    return c_map


class SemanticPredMaskRCNN():

    def __init__(self, args):
        self.segmentation_model = ImageSegmentation(args)
        if args.segformer:
            self.device = torch.device("cuda:" + str(args.sem_gpu_id))

            model_path = "Stubborn/agent/utils/segformer-b4-finetuned-ade-512-512"
            self.feature_extractor = SegformerFeatureExtractor.from_pretrained(model_path)
            self.segformer = SegformerForSemanticSegmentation.from_pretrained(model_path)
            self.segformer.to(self.device)
            self.segformer.eval()
        self.args = args

    def get_prediction(self, img, depth=None, goal_cat=None):
        args = self.args
        
        if args.segformer:
            with torch.no_grad():
                pixel_values = self.feature_extractor(img, return_tensors="pt").pixel_values.to(self.device)
                outputs = self.segformer(pixel_values)
                gc = hm3d_to_ade[coco_to_hm3d[goal_cat]]
                smx = outputs.logits.softmax(dim=1)[0, gc]
                logits = nn.functional.interpolate(outputs.logits.detach().cpu()[:, gc:gc+1],
                    size=(480, 640), # (height, width)
                    mode='bilinear',
                    align_corners=False)
                msk = logits > float(args.sf_thr)
                if goal_cat == 3:  # bed vs sofa
                    oc = hm3d_to_ade[coco_to_hm3d[1]]
                    otherlogits = nn.functional.interpolate(outputs.logits.detach().cpu()[:, oc:oc+1],
                        size=(480, 640), # (height, width)
                        mode='bilinear',
                        align_corners=False)
                    msk *= logits > otherlogits
                if goal_cat == 1:  # sofa vs chair
                    oc = hm3d_to_ade[coco_to_hm3d[0]]
                    otherlogits = nn.functional.interpolate(outputs.logits.detach().cpu()[:, oc:oc+1],
                        size=(480, 640), # (height, width)
                        mode='bilinear',
                        align_corners=False)
                    msk *= logits > otherlogits
    
        image_list = []
        img = img[:, :, ::-1]
        
        image_list.append(img)
        seg_predictions, vis_output = self.segmentation_model.get_predictions(
            image_list, visualize=args.visualize == 2)
        
        

        if args.visualize == 2:
            img = vis_output.get_image()

        semantic_input = np.zeros((img.shape[0], img.shape[1], 15 + 1))
        high_thr = 0.9
        for j, class_idx in enumerate(
                seg_predictions[0]['instances'].pred_classes.cpu().numpy()):
            if class_idx in list(coco_categories_mapping.keys()):
                idx = coco_categories_mapping[class_idx]
                confscore = seg_predictions[0]['instances'].scores[j]
                if (confscore < high_thr and (idx not in [5])) or (confscore < args.sem_pred_prob_thr and (idx in [5])):
                    continue
                else:
                    obj_mask = seg_predictions[0]['instances'].pred_masks[j] * 1.
                    semantic_input[:, :, idx] += obj_mask.cpu().numpy()

        
        semantic_input[:, :, 3] *= semantic_input[:, :, 1] < high_thr
        semantic_input[:, :, 1] *= semantic_input[:, :, 0] < high_thr
        
        if args.segformer:
            semantic_input[:, :, goal_cat] = msk
        return semantic_input, img


def compress_sem_map(sem_map):
    c_map = np.zeros((sem_map.shape[1], sem_map.shape[2]))
    for i in range(sem_map.shape[0]):
        c_map[sem_map[i] > 0.] = i + 1
    return c_map


class ImageSegmentation():
    def __init__(self, args):
        string_args = """
            --config-file Stubborn/agent/utils/COCO-InstSeg/mask_rcnn_R_101_FPN_3x.yaml
            --input input1.jpeg
            --confidence-threshold {}
            --opts MODEL.WEIGHTS
            detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl
            
            """.format(args.sem_pred_prob_thr)
            # detectron2://new_baselines/mask_rcnn_R_101_FPN_400ep_LSJ/42073830/model_final_f96b26.pkl
            # COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
            
        if args.sem_gpu_id == -2:
            string_args += """ MODEL.DEVICE cpu"""
        else:
            string_args += """ MODEL.DEVICE cuda:{}""".format(args.sem_gpu_id)

        string_args = string_args.split()

        args = get_seg_parser().parse_args(string_args)
        logger = setup_logger()
        logger.info("Arguments: " + str(args))

        cfg = setup_cfg(args)
        self.demo = VisualizationDemo(cfg)

    def get_predictions(self, img, visualize=0):
        return self.demo.run_on_image(img, visualize=visualize)


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = \
        args.confidence_threshold
    cfg.freeze()
    
    # cfg = get_config("new_baselines/mask_rcnn_R_50_FPN_400ep_LSJ.py")
    # print(LazyConfig.to_py(cfg))
    # cfg.train.init_checkpoint = "detectron2://new_baselines/mask_rcnn_R_50_FPN_400ep_LSJ/42019571/model_final_14d201.pkl"

    
    return cfg


def get_seg_parser():
    parser = argparse.ArgumentParser(
        description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
        """
        # self.metadata = MetadataCatalog.get(
        #     cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        # )
        self.metadata = MetadataCatalog.get("coco_2017_val")
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.predictor = BatchPredictor(cfg)

    def run_on_image(self, image_list, visualize=0):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        all_predictions = self.predictor(image_list)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.

        if visualize:
            predictions = all_predictions[0]
            image = image_list[0]
            visualizer = Visualizer(
                image, self.metadata, instance_mode=self.instance_mode)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_output = visualizer.draw_panoptic_seg_predictions(
                    panoptic_seg.to(self.cpu_device), segments_info
                )
            else:
                if "sem_seg" in predictions:
                    vis_output = visualizer.draw_sem_seg(
                        predictions["sem_seg"].argmax(
                            dim=0).to(self.cpu_device)
                    )
                if "instances" in predictions:
                    instances = predictions["instances"].to(self.cpu_device)
                    vis_output = visualizer.draw_instance_predictions(
                        predictions=instances)

        return all_predictions, vis_output


class BatchPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a list of input images.
    Compared to using the model directly, this class does the following
    additions:
    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by
         `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take a list of input images
    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained
            from cfg.DATASETS.TEST.
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        # self.cfg = cfg
        
        self.model = build_model(self.cfg)
        # self.model = instantiate(cfg.model)
        # self.model.to(cfg.train.device)
        # self.model = create_ddp_model(self.model)
        
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        #checkpointer.load(cfg.train.init_checkpoint)
        self.model.eval()
        
        self.input_format = 'BGR' #cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, image_list):
        """
        Args:
            image_list (list of np.ndarray): a list of images of
                                             shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):
                the output of the model for all images.
                See :doc:`/tutorials/models` for details about the format.
        """
        inputs = []
        for original_image in image_list:
            # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = original_image
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            instance = {"image": image, "height": height, "width": width}

            inputs.append(instance)

        with torch.no_grad():
            predictions = self.model(inputs)
            return predictions
        