import os, cv2, random, argparse, json

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

from detectron2.utils.logger import setup_logger
setup_logger()

parser = argparse.ArgumentParser()
parser.add_argument('--type', default='control')
parser.add_argument('--models', default='O_n')
parser.add_argument('--model', default='O10')
parser.add_argument('--BoL', default='best')
parser.add_argument('--output', type=str, default='quantitative', help='output folder')  # output folder
args = parser.parse_args()

test_images = "../datasets/T/test"
test_annos = "../datasets/T/annotations/test.json"

#fix_annotations(test_annos)
# TEST SET
register_coco_instances("corrosion_test", {}, test_annos, test_images)

out_dir = os.path.join(args.output, args.models, args.model)
os.makedirs(out_dir,exist_ok=True)
cfg = get_cfg()
cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = os.path.join('models',args.type,args.models,args.model,f'{args.BoL}.pth')
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05   # set the testing threshold for this model
cfg.DATASETS.TEST = ("corrosion_test", )
predictor = DefaultPredictor(cfg)

val_loader = build_detection_test_loader(cfg, "corrosion_test")
evaluator = COCOEvaluator("corrosion_test", cfg, False, output_dir=args.output)

print(f"Evaluating {args.models}/{args.model}")
res = inference_on_dataset(predictor.model, val_loader, evaluator)
with open(os.path.join(out_dir, 'results.txt'), 'w') as json_file:
    json.dump(dict(res), json_file)
