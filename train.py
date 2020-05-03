import torch
import time
import datetime
import logging

# import detectron2 utilities
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context, COCOEvaluator
from detectron2.utils.logger import log_every_n_seconds, setup_logger
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm

class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
    
    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
            
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):            
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses
            
    def _get_loss(self, data):
        # How loss is calculated on train_loop 
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
        
        
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
                     
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks

if __name__=='__main__':
    # import common libraries
    import os, argparse, math
    import numpy as np
    import cv2
    import random
    from change_ckpt import change_ckpt_iter

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='path to training set')
    parser.add_argument('--epochs', help='epochs to train')
    parser.add_argument('--weights', help='weights to use during training')
    parser.add_argument('--out_dir', help='output directory')
    parser.add_argument('--resume', help='flag to resume training', default=False, type=bool)
    parser.add_argument('--bs', help='batch size', default=1)

    args = parser.parse_args()

    setup_logger()

    # Datasets
    train_annos = os.path.join("../datasets", args.train, 'annotations/train.json')
    train_images= os.path.join("../datasets", args.train, 'train')
    val_annos = "../datasets/V/annotations/val.json"
    val_images = "../datasets/V/val"

    # Registering coco datasets
    register_coco_instances("corrosion_train", {}, train_annos, train_images)
    register_coco_instances("corrosion_val", {}, val_annos, val_images)

    
    num_images = len(os.listdir(train_images))
    print(num_images)
    # Train
    cfg = get_cfg()
    cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("corrosion_train",)
    cfg.DATASETS.TEST = ("corrosion_val",)
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.MODEL.WEIGHTS = (args.weights)
    cfg.SOLVER.IMS_PER_BATCH = args.bs
    #cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.CHECKPOINT_PERIOD = num_images
    cfg.SOLVER.MAX_ITER = num_images * int(args.epochs)
    cfg.TEST.EVAL_PERIOD = math.floor(num_images / 20) * 20
    print(cfg.TEST.EVAL_PERIOD)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (corrosion)
    cfg.OUTPUT_DIR = args.out_dir

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = MyTrainer(cfg) 
    trainer.resume_or_load(resume=args.resume)
    trainer.train()

    change_ckpt_iter(os.path.join(cfg.OUTPUT_DIR, 'model_final.pth'), os.path.join(cfg.OUTPUT_DIR, 'last.pth'), 0)

    