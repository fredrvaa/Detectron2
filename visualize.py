import os, cv2, random, argparse

from detectron2.modeling import build_model
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.data.datasets.coco import load_coco_json
from detectron2.evaluation import inference_on_dataset
from detectron2.evaluation import COCOEvaluator

from fix_annotations import fix_annotations

print('Initialized')

# TRAIN SET
register_coco_instances("train", {}, "../datasets/D10/annotations/train.json", "../datasets/D10/train")

# VAL SET
register_coco_instances("train", {}, "../datasets/D10/annotations/train.json", "../datasets/D10/train")

# Visualize
# for i,d in enumerate(random.sample(dataset_dicts, 10)):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=dataset_metadata, scale=0.5)
#     vis = visualizer.draw_dataset_dict(d)
#     cv2.imshow(f"images/image_{i}", vis.get_image()[:, :, ::-1])
#     cv2.waitKey(-1)


model = "/media/fredrik/HDD/Master/models/Faster R-CNN/150k[base]/model_0149999.pth"
cfg = get_cfg()
cfg.OUTPUT_DIR = out_dir
cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.DATASETS.TEST = ("corrosion_train", )
predictor = DefaultPredictor(cfg)

dataset_dicts = DatasetCatalog.get("train")
dataset_metadata = MetadataCatalog.get("train")

# # Predictions
# for i,d in enumerate(random.sample(dataset_dicts, 10)):
#     print(i)
#     im = cv2.imread(d["file_name"])
#     outputs = predictor(im)
#     v = Visualizer(im[:, :, ::-1],
#                    metadata=dataset_metadata)
#     v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     cv2.imwrite(f"pred_images/image_{i}.jpg", v.get_image()[:, :, ::-1])
