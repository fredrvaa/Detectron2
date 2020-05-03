import os, cv2, random, argparse

from detectron2.modeling import build_model
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.data.datasets.coco import load_coco_json
from detectron2.evaluation import inference_on_dataset
from detectron2.evaluation import COCOEvaluator

from fix_annotations import fix_annotations
from datasets import Base, BaseNew

print('Initialized')


fix_annotations()

# TRAIN SET
register_coco_instances("corrosion_train", {}, dataset.train_annos, dataset.train_images)

# TEST SET
register_coco_instances("corrosion_test", {}, dataset.test_annos, dataset.test_images)

#AUGMENTED TEST SET
register_coco_instances("corrosion_aug", {}, dataset.aug_annos, dataset.aug_images)

#out_dir = '/media/fredrik/HDD/Master/models/Faster R-CNN/150k[base]'               
#out_dir = '/media/fredrik/HDD/Master/models/Faster R-CNN/150k[base+new]'           
#out_dir = '/media/fredrik/HDD/Master/models/Faster R-CNN/150k[base+new]2'          
out_dir = '/media/fredrik/HDD/Master/models/Faster R-CNN/100k[base]_50k[base+new]'

'''
|----------------TEST-----------------|-----------------AUG-----------------|
|mAP@.5 = 0.742 | mAP@[.5:.95] = 0.470|mAP@.5 = 0.695 | mAP@[.5:.95] = 0.423|
|mAP@.5 = 0.765 | mAP@[.5:.95] = 0.475|mAP@.5 = 0.678 | mAP@[.5:.95] = 0.412|
|mAP@.5 = 0.760 | mAP@[.5:.95] = 0.480|mAP@.5 = 0.682 | mAP@[.5:.95] = 0.418|
|mAP@.5 = 0.741 | mAP@[.5:.95] = 0.463|mAP@.5 = 0.742 | mAP@[.5:.95] = 0.470|
'''

with open(os.path.join(out_dir, "last_checkpoint")) as file:
    model = file.readline()

model = "/media/fredrik/HDD/Master/models/Faster R-CNN/150k[base]/model_0149999.pth"

cfg = get_cfg()
cfg.OUTPUT_DIR = out_dir
cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.DATASETS.TEST = ("corrosion_train", )
predictor = DefaultPredictor(cfg)

val_loader = build_detection_test_loader(cfg, "corrosion_train")
evaluator = COCOEvaluator("corrosion_train", cfg, False, output_dir=out_dir)

inference_on_dataset(predictor.model, val_loader, evaluator)

