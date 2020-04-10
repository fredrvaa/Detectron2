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

dataset = BaseNew()

fix_annotations(dataset.annotation_dir)

# Registering coco datasets
register_coco_instances("corrosion_test", {}, dataset.test_annos, dataset.test_images)

with open(os.path.join(dataset.save_dir, "last_checkpoint")) as file:
    model = file.readline()

model = "model_0047999.pth"

cfg = get_cfg()
cfg.OUTPUT_DIR = dataset.save_dir
cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.DATASETS.TEST = ("corrosion_test", )
predictor = DefaultPredictor(cfg)

dataset_dicts = load_coco_json(dataset.test_annos, dataset.test_images, "corrosion_test")
dataset_metadata = MetadataCatalog.get("corrosion_test")

model = build_model(cfg)
val_loader = build_detection_test_loader(cfg, "corrosion_test")
evaluator = COCOEvaluator("corrosion_test", cfg, False, output_dir=dataset.save_dir)

inference_on_dataset(predictor.model, val_loader, evaluator)

# for d in random.sample(dataset_dicts, 10):
#     im = cv2.imread(d["file_name"])
#     outputs = predictor(im)
#     v = Visualizer(im[:, :, ::-1],
#                    metadata=dataset_metadata,
#                    scale=0.3
#     )
#     v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     cv2.imshow('predicted', v.get_image()[:, :, ::-1])
#     cv2.waitKey(-1)