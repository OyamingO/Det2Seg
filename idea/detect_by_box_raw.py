import os
import cv2
import torch
import argparse
import numpy as np
from pathlib import Path
from segment_anything import SamPredictor, sam_model_registry


def multi_box_prompt(draw_path, predictor, boxes, image):
    masks_in_curr_img = []
    for i_box in range(len(boxes)):
        cur_box = torch.tensor(boxes[i_box], device=predictor.device)	
        transformed_boxes = predictor.transform.apply_boxes_torch(cur_box, image.shape[:2])
        masks_in_curr_box, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        masks_in_curr_box = masks_in_curr_box.squeeze(0).squeeze(0)
        masks_in_curr_img.append(masks_in_curr_box)
    return masks_in_curr_img
   
sam = sam_model_registry[args.model_type](checkpoint=args.weights).to(device)
predictor = SamPredictor(sam)
predictor.set_image(img)
masks = multi_box_prompt(draw_path, predictor, boxes, img)
            