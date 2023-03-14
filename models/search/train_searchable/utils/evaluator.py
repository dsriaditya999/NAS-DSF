import os
import abc
import json
from tempfile import NamedTemporaryFile

import numpy as np
import torch
import torch.distributed as dist

from pycocotools.cocoeval import COCOeval
from effdet.distributed import synchronize, is_main_process, all_gather_container


class Evaluator:

    def __init__(self, distributed=False, pred_yxyx=False):
        self.distributed = distributed
        self.distributed_device = None
        self.pred_yxyx = pred_yxyx
        self.img_indices = []
        self.predictions = []

    def add_predictions(self, detections, target):
        if self.distributed:
            if self.distributed_device is None:
                # cache for use later to broadcast end metric
                self.distributed_device = detections.device
            synchronize()
            detections = all_gather_container(detections)
            img_indices = all_gather_container(target['img_idx'])
            if not is_main_process():
                return
        else:
            img_indices = target['img_idx']

        detections = detections.cpu().numpy()
        img_indices = img_indices.cpu().numpy()
        for img_idx, img_dets in zip(img_indices, detections):
            self.img_indices.append(img_idx)
            self.predictions.append(img_dets)

    def _coco_predictions(self):
        # generate coco-style predictions
        coco_predictions = []
        coco_ids = []
        for img_idx, img_dets in zip(self.img_indices, self.predictions):
            img_id = self._dataset.img_ids[img_idx]
            coco_ids.append(img_id)
            if self.pred_yxyx:
                # to xyxy
                img_dets[:, 0:4] = img_dets[:, [1, 0, 3, 2]]
            # to xywh
            img_dets[:, 2] -= img_dets[:, 0]
            img_dets[:, 3] -= img_dets[:, 1]
            for det in img_dets:
                score = float(det[4])
                if score < .001:  # stop when below this threshold, scores in descending order
                    break
                coco_det = dict(
                    image_id=img_id,
                    bbox=det[0:4].tolist(),
                    score=score,
                    category_id=int(det[5]))
                coco_predictions.append(coco_det)
        return coco_predictions, coco_ids

    @abc.abstractmethod
    def evaluate(self, output_result_file=''):
        pass

    def save(self, result_file):
        # save results in coco style, override to save in a alternate form
        if not self.distributed or dist.get_rank() == 0:
            assert len(self.predictions)
            coco_predictions, coco_ids = self._coco_predictions()
            json.dump(coco_predictions, open(result_file, 'w'), indent=4)


class CocoEvaluator(Evaluator):

    def __init__(self, dataset, distributed=False, pred_yxyx=False):
        super().__init__(distributed=distributed, pred_yxyx=pred_yxyx)
        self._dataset = dataset.parser
        self.coco_api = dataset.parser.coco

    def reset(self):
        self.img_indices = []
        self.predictions = []

    def evaluate(self, output_result_file=''):
        if not self.distributed or dist.get_rank() == 0:
            assert len(self.predictions)
            coco_predictions, coco_ids = self._coco_predictions()
            if output_result_file:
                json.dump(coco_predictions, open(output_result_file, 'w'), indent=4)
                results = self.coco_api.loadRes(output_result_file)
            else:
                with NamedTemporaryFile(prefix='coco_', suffix='.json', delete=False, mode='w') as tmpfile:
                    json.dump(coco_predictions, tmpfile, indent=4)
                results = self.coco_api.loadRes(tmpfile.name)
                try:
                    os.unlink(tmpfile.name)
                except OSError:
                    pass
            coco_eval = COCOeval(self.coco_api, results, 'bbox')
            coco_eval.params.imgIds = coco_ids  # score only ids we've used
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            metric = coco_eval.stats[0]  # mAP 0.5-0.95
            if self.distributed:
                dist.broadcast(torch.tensor(metric, device=self.distributed_device), 0)
        else:
            metric = torch.tensor(0, device=self.distributed_device)
            dist.broadcast(metric, 0)
            metric = metric.item()
        self.reset()
        return metric