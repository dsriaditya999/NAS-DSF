import copy

import ensemble_boxes
import numpy as np
import torch
import torchvision
from PIL import Image



def normalize_boxes(boxes, img_size, invert=False):
    if invert:
        boxes[..., 0] = boxes[..., 0] * img_size[0]
        boxes[..., 1] = boxes[..., 1] * img_size[1]
        boxes[..., 2] = boxes[..., 2] * img_size[0]
        boxes[..., 3] = boxes[..., 3] * img_size[1]
    else:
        boxes[..., 0] = boxes[..., 0] / img_size[0]
        boxes[..., 1] = boxes[..., 1] / img_size[1]
        boxes[..., 2] = boxes[..., 2] / img_size[0]
        boxes[..., 3] = boxes[..., 3] / img_size[1]
    return boxes


def detections_to_self_label(detections1, detections2, target, score_threshold=0.1, use_weighted_boxes_fusion=False):
    self_label = copy.deepcopy(target)
    for idx, [img_dets1, img_dets2] in enumerate(zip(detections1, detections2)):
        
        # boxes fusion
        if use_weighted_boxes_fusion:
            aggregate_detections = torch.cat((torch.unsqueeze(img_dets1, 0), torch.unsqueeze(img_dets2, 0)), dim=0)
            aggregate_detections = normalize_boxes(aggregate_detections, self_label['img_size'][idx])

            boxes, scores, labels = ensemble_boxes.weighted_boxes_fusion(
                aggregate_detections[:, :, 0:4], 
                aggregate_detections[:, :, 4], 
                aggregate_detections[:, :, 5])
            
            boxes = torch.from_numpy(boxes).to(aggregate_detections.device)
            scores = torch.from_numpy(scores).to(aggregate_detections.device)
            labels = torch.from_numpy(labels).to(aggregate_detections.device)
            boxes = normalize_boxes(boxes, self_label['img_size'][idx], invert=True)
        else:
            aggregate_detections = torch.cat((img_dets1, img_dets2), dim=0)
            idx2keep = torchvision.ops.nms(aggregate_detections[:,:4], aggregate_detections[:,4], iou_threshold=0.5)

            processed_detections = aggregate_detections[idx2keep]
            boxes = processed_detections[:,0:4]
            scores = processed_detections[:,4]
            labels = processed_detections[:,5]

        # from xyxy to yxyx
        boxes[:, 0:4] = boxes[:, [1, 0, 3, 2]] / self_label['img_scale'][idx]
        # score filter

        valid_idx = torch.where(torch.logical_and((scores > score_threshold), (labels < 4)))[0]
        boxes = torch.index_select(boxes, 0, valid_idx)
        labels = torch.index_select(labels, 0, valid_idx)
        scores = torch.index_select(scores, 0, valid_idx)
        
        self_label['scores'] = 0*self_label['cls'][:] - 1
        padding_len = self_label['bbox'][idx].shape[0] - len(valid_idx)
        if padding_len > 0:
            padding = -1 * torch.ones((padding_len, 6)).to(aggregate_detections.device)
            self_label['bbox'][idx] = torch.cat((boxes, padding[:, 0:4]), dim=0)
            self_label['cls'][idx] = torch.cat((labels, padding[:, 4]), dim=0)
            self_label['scores'][idx] = torch.cat((scores, padding[:, 5]), dim=0)
        elif padding_len <= 0:
            self_label['bbox'][idx] = boxes[0: self_label['bbox'][idx].shape[0], :]
            self_label['cls'][idx] = labels[0: self_label['bbox'][idx].shape[0], :]
            self_label['scores'][idx] = scores[0: self_label['bbox'][idx].shape[0], :]

    return self_label


def bounding_boxes(v_boxes, v_labels, v_scores, log_width, log_height, class_id_to_label, score_threshold):
    all_boxes = []
    # plot each bounding box for this image
    for b_i, box in enumerate(v_boxes):
        # get coordinates and labels

        if v_scores[b_i] < score_threshold or int(v_labels[b_i]) == -1:  # stop when below this threshold, scores in descending order
            break

        caption = "%s" % (class_id_to_label[int(v_labels[b_i])])
        if v_scores[b_i] <= 1:
            caption = "%s (%.3f)" % (class_id_to_label[int(v_labels[b_i])], v_scores[b_i])
        # from xyxy
        box_data = {"position" : {
            "minX" : int(box[0]),
            "minY" : int(box[1]),
            "maxX" : int(box[2]),
            "maxY" : int(box[3])},
            "class_id" : int(v_labels[b_i]),
            # optionally caption each box with its class and score
            "box_caption" : caption,
            "domain" : "pixel",
            "scores" : { "score" : int(v_scores[b_i]*100) }}

        all_boxes.append(box_data)
    return all_boxes


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.clamp(-1.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def visualize_detections(dataset, detections, target, wandb, args, split='val', score_threshold=0.5, img_tensor=None):
    class_id_to_label = { int(i) : str(i) for i in range(1, args.num_classes + 1)}
    class_id_to_label.update({1: "person", 2: "bicycle", 3: "car"})
    detections = detections.detach().cpu().numpy()
    img_indices = target['img_idx'].cpu().numpy()
    bboxes = target['bbox'].cpu().numpy()
    clses = target['cls'].cpu().numpy()
    scores = target['scores'].cpu().numpy() if 'scores' in target else np.zeros_like(clses) + 1000
    img_scales = target['img_scale'].cpu().numpy()
    for i, (img_idx, img_dets, bbox, cls, img_scale, score) in enumerate(zip(img_indices, detections, bboxes, clses, img_scales, scores)):
        img_id = dataset.parser.img_ids[img_idx]
        img_info = dataset.parser.img_infos[img_idx]
        # yxyx to xyxy
        if img_tensor is None:
            bbox[:, 0:4] = bbox[:, [1, 0, 3, 2]] * img_scale
            filename = dataset.thermal_data_dir/img_info['file_name']
            raw_image = Image.open(filename).convert('RGB')
        else:
            bbox[:, 0:4] = bbox[:, [1, 0, 3, 2]]
            img_dets[:, 0:4] = img_dets[:, 0:4] / img_scale
            raw_image = tensor2im(img_tensor[i])
        predicted_boxes = bounding_boxes(
            v_boxes=img_dets[:, 0:4],
            v_labels=img_dets[:, 5],
            v_scores=img_dets[:, 4],
            log_width=img_info['width'],
            log_height=img_info['height'],
            class_id_to_label=class_id_to_label,
            score_threshold=score_threshold)
        gt_boxes = bounding_boxes(
            v_boxes=bbox,
            v_labels=cls,
            v_scores=score,
            log_width=img_info['width'],
            log_height=img_info['height'],
            class_id_to_label=class_id_to_label,
            score_threshold=0)
        # log to wandb: raw image, predictions, and dictionary of class labels for each class id
        box_image = wandb.Image(raw_image, boxes = {"predictions": {"box_data": predicted_boxes, "class_labels" : class_id_to_label},
                                                    "gts": {"box_data": gt_boxes, "class_labels" : class_id_to_label}})
        wandb.log({split: box_image})


def visualize_target(dataset, target, wandb, args, split='val', img_tensor=None):
    class_id_to_label = { int(i) : str(i) for i in range(1, args.num_classes + 1)}
    class_id_to_label.update({1: "person", 2: "bicycle", 3: "car"})
    img_indices = target['img_idx'].cpu().numpy()
    bboxes = target['bbox'].cpu().numpy()
    clses = target['cls'].cpu().numpy()
    scores = target['scores'].cpu().numpy() if 'scores' in target else np.zeros_like(clses) + 1000
    img_scales = target['img_scale'].cpu().numpy()
    for i, (img_idx, bbox, cls, img_scale, score) in enumerate(zip(img_indices, bboxes, clses, img_scales, scores)):
        img_id = dataset.parser.img_ids[img_idx]
        img_info = dataset.parser.img_infos[img_idx]
        # yxyx to xyxy
        if img_tensor is None:
            bbox[:, 0:4] = bbox[:, [1, 0, 3, 2]] * img_scale
            filename = dataset.thermal_data_dir/img_info['file_name']
            raw_image = Image.open(filename).convert('RGB')
        else:
            bbox[:, 0:4] = bbox[:, [1, 0, 3, 2]]
            raw_image = tensor2im(img_tensor[i])
        gt_boxes = bounding_boxes(
            v_boxes=bbox,
            v_labels=cls,
            v_scores=score,
            log_width=img_info['width'],
            log_height=img_info['height'],
            class_id_to_label=class_id_to_label,
            score_threshold=0)
        # log to wandb: raw image, predictions, and dictionary of class labels for each class id
        box_image = wandb.Image(raw_image, boxes = {'gts': {"box_data": gt_boxes, "class_labels" : class_id_to_label}})
        wandb.log({split: box_image})
