#!/usr/bin/env python
"""
A main training script.
This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.
In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".
Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
import pickle
import os.path as osp
import numpy as np
import pycocotools
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.structures import BoxMode


categories = ['chair', 'sofa', 'plant', 'bed', 'toilet', 'tv_monitor',  
              'fireplace', 'bathtub', 'mirror'] #'cabinet', 'sink', 'cushion', 'chest_of_drawers']
ASPECT_RATIO_THRESH = 10
MASK_AREA_THRESH = 1000


def hm3d_seg_dataset_fn(base_dir):
    out = []
    for img_file in os.listdir(osp.join(base_dir, 'rgb')):
        d = {}
        d['file_name'] = osp.join(base_dir, 'rgb', img_file)
        d['height'] = 480
        d['width'] = 640
        d['image_id'] = img_file[:-4]
        annots = pickle.load(open(osp.join(base_dir, 'sem', d['image_id'] + '.pkl'), 'rb'))
        for i in range(len(annots) - 1, -1, -1):
            an = annots[i]
            if an['cat'] not in categories:
                annots.pop(i)
                continue
                
            yy, xx = an['idxs']
            if len(xx) < MASK_AREA_THRESH:
                annots.pop(i)
                continue
            bbox = [np.min(xx), np.min(yy), np.max(xx), np.max(yy)]
            bbh, bbw = bbox[3] - bbox[1] + 1, bbox[2] - bbox[0] + 1
            if max(bbh/bbw, bbw/bbh) > ASPECT_RATIO_THRESH:
                annots.pop(i)
                continue
                
            an['bbox'] = bbox
            an['bbox_mode'] = BoxMode.XYXY_ABS
            an['category_id'] = categories.index(an['cat'])
            msk = np.zeros((480, 640), dtype=np.uint8)
            msk[an['idxs']] = 1
            an['segmentation'] = pycocotools.mask.encode(np.asarray(msk, order="F"))
        d['annotations'] = annots
        out.append(d)
    return out


def hm3dseg_val_dataset_fn():
    return hm3d_seg_dataset_fn('data/seg/val')

def hm3dseg_train_dataset_fn():
    return hm3d_seg_dataset_fn('data/seg/train')


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    #"detectron2://new_baselines/mask_rcnn_R_101_FPN_400ep_LSJ/42073830/model_final_f96b26.pkl" 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(categories)
    cfg.MODEL.RETINANET.NUM_CLASSES = len(categories)
    #cfg.MODEL.BACKBONE.FREEZE_AT = 5
    cfg.SOLVER.IMS_PER_BATCH = 16
    cfg.SOLVER.CHECKPOINT_PERIOD = 5000
    cfg.SOLVER.STEPS = 20000, 40000
    cfg.SOLVER.MAX_ITER = 50000
    cfg.DATASETS.TRAIN = ('hm3dseg_train',)
    cfg.DATASETS.TEST = ('hm3dseg_val',)
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.INPUT.FORMAT = 'RGB'
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    DatasetCatalog.register('hm3dseg_train', hm3dseg_train_dataset_fn)
    DatasetCatalog.register('hm3dseg_val', hm3dseg_val_dataset_fn)
    MetadataCatalog.get("hm3dseg_train").thing_classes = categories
    MetadataCatalog.get("hm3dseg_train").evaluator_type = 'coco'
    MetadataCatalog.get("hm3dseg_val").thing_classes = categories
    MetadataCatalog.get("hm3dseg_val").evaluator_type = 'coco'
    
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )