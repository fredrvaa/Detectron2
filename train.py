
if __name__=='__main__':
    # import common libraries
    import os, argparse
    import numpy as np
    import cv2
    import random

    from fix_annotations import fix_annotations

    # import detectron2 utilities
    from detectron2.engine import DefaultTrainer
    from detectron2.config import get_cfg
    from detectron2.data.datasets import register_coco_instances
    from detectron2.utils.logger import setup_logger
    setup_logger()

    import datasets
    dataset = datasets.BaseNew()

    fix_annotations(dataset.annotation_dir)

    # Registering coco datasets
    register_coco_instances("corrosion_train", {}, dataset.train_annos, dataset.train_images)
    register_coco_instances("corrosion_val", {}, dataset.val_annos, dataset.val_images)

    # Train
    cfg = get_cfg()
    cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("corrosion_train",)
    cfg.DATASETS.TEST = ("corrosion_val")
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.MODEL.WEIGHTS = ('/media/fredrik/HDD/Master/models/Faster R-CNN/150k[base]/model_0098999.pth')
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.CHECKPOINT_PERIOD = 3000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (corrosion)
    cfg.OUTPUT_DIR = '/media/fredrik/HDD/Master/models/Faster R-CNN/100k[base]_50k[base+new]'

    # for d in random.sample(dataset_dicts, 10):
    #     img = cv2.imread(d["file_name"])
    #     visualizer = Visualizer(img[:, :, ::-1], metadata=dataset_metadata, scale=0.3)
    #     vis = visualizer.draw_dataset_dict(d)
    #     cv2.imshow('image', vis.get_image()[:, :, ::-1])
    #     cv2.waitKey(-1)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=True)
    trainer.train()
