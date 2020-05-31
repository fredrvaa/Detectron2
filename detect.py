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

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c1 = c1[0], c2[1] + t_size[1]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights')
    parser.add_argument('--output', type=str, default='output2', help='output folder')  # output folder
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    test_images = "../datasets/paint_flake"

    cfg = get_cfg()
    cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3   # set the testing threshold for this model
    predictor = DefaultPredictor(cfg)

    # Predictions
    for file_name in os.listdir(test_images):
        print(file_name)
        im = cv2.imread(os.path.join(test_images, file_name))
        outputs = predictor(im)['instances']
        for bbox,score in zip(outputs.get('pred_boxes').tensor,outputs.get('scores')):
            s = round(float(score.tolist()),2)
            plot_one_box(bbox,im,color=(0,0,255),label=(f"corrosion {s}"))

        cv2.imwrite(f"{args.output}/{file_name}", im)
