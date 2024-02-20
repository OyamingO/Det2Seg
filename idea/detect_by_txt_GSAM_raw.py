import os
import cv2
import torch
import argparse
import numpy as np
from pathlib import Path
from segment_anything import build_sam, SamPredictor, build_sam_hq

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


def multi_box_prompt(args, draw_path, boxes, image):
    masks_in_curr_img = []
    for i_box in range(len(boxes)):
        cur_box = torch.tensor(boxes[i_box][1:], device=args.device)
        transformed_boxes = args.predictor.transform.apply_boxes_torch(cur_box, image.shape[:2])
        masks_in_curr_box, _, _ = args.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        masks_in_curr_box = masks_in_curr_box.squeeze(0).squeeze(0)
        masks_in_curr_img.append(masks_in_curr_box)
    return masks_in_curr_img


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases



# load grounding dino model
model = load_model(config_file, grounded_checkpoint, device=device)
# run grounding dino model
boxes_filt, pred_phrases = get_grounding_output(model, image, args.obj_class, box_threshold, text_threshold, device=device)

# build sam
predictor = SamPredictor(build_sam(checkpoint=weights).to(device))
predictor.set_image(original_img)
# run SAM
masks = multi_box_prompt(args, draw_path, boxes, original_img)
   