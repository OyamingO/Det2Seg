import os
import cv2
import torch
import argparse
import numpy as np
from pathlib import Path
from segment_anything import SamPredictor, sam_model_registry
from HC_saliency import *


def generate_random_negative_points_in_img(args, background_area):
    true_count = torch.count_nonzero(background_area)  
    num_neg_points = int(true_count * 0.01)  
    negtive_points = get_random_XY_pts(args, background_area, num_neg_points)
    return negtive_points


def get_random_XY_pts(args, mask_area, num_points):
    if not torch.is_tensor(mask_area):
        mask_area = torch.from_numpy(mask_area).float().to(args.device)
    selected_pts = []
    true_pts = torch.nonzero(mask_area).cpu().numpy()  
    if len(true_pts)!=0 and num_points: 
        if len(true_pts) <= num_points:
            selected_pts = true_pts
        else:
            selected_pts = true_pts[np.random.choice(len(true_pts), num_points, replace=False)]
    else:
        mask_area.fill_(True)
        selected_pts = torch.nonzero(mask_area).cpu().numpy()
        
    selected_pts[:, [0, 1]] = selected_pts[:, [1, 0]] 
    return selected_pts


def generate_random_positive_points_HC(args, buffer_img, box_in_buffer, draw_path, i_box):
    category_id, x1, y1, x2, y2 = box_in_buffer
    w, h = x2 - x1, y2 - y1
    bbox_area = buffer_img[y1:y2, x1:x2]
    if w < 32 and h < 32: 
        region_mask = torch.ones((bbox_area.shape[0], bbox_area.shape[1]), dtype=bool, device=args.device) 
        saliency_points = get_random_XY_pts(args, region_mask, args.num_pos_pts)
        return torch.tensor(saliency_points, device=args.device).float() 

    # HC saliency
    img_float = get_img3_float(bbox_area)
    saliency_map = GetHC(img_float)
    saliency_gray = (saliency_map * 255).astype(np.uint8)

    threshold, saliency_mask = cv2.threshold(saliency_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    saliency_mask = saliency_mask.astype(bool)
    saliency_points = get_random_XY_pts(args, saliency_mask, args.num_pos_pts)
    
    saliency_points[:, 0] += int(x1)  
    saliency_points[:, 1] += int(y1)  
    return torch.tensor(saliency_points, device=args.device).float()


def multi_point_prompt(args, boxes, image, background_area, draw_path):    
    masks_in_curr_img = []
    mask_box = []
    negative_points_in_img = generate_random_negative_points_in_img(args, background_area)
    for i_box in range(len(boxes)):
        prompt_points_dict = {category_id: [] for category_id in range(len(args.categories))}
        prompt_points_dict[-1] = []  

        negative_points_in_img_buffer_coor = np.zeros_like(negative_points_in_img)
        cur_box = torch.tensor(boxes[i_box], device=args.device)
        buffer_img, buffer_mask, buffer_bbox = add_buffer_on_box(image, cur_box, draw_path, i_box)  
        args.predictor.set_image(buffer_img)
        cur_box[1] -= buffer_bbox[0]
        cur_box[2] -= buffer_bbox[1]
        cur_box[3] -= buffer_bbox[0]
        cur_box[4] -= buffer_bbox[1]
        if len(negative_points_in_img) != 0:
            negative_points_in_img_buffer_coor[:, 0] = negative_points_in_img[:, 0] - int(buffer_bbox[0]) 
            negative_points_in_img_buffer_coor[:, 1] = negative_points_in_img[:, 1] - int(buffer_bbox[1]) 
        negative_points = []
        for point in negative_points_in_img_buffer_coor:
            if len(negative_points) > args.num_neg_pts:
                break
            if point[1]>=0 and point[0]>=0:
                try:
                    if buffer_mask[point[1]][point[0]] == True:
                        negative_points.append(point)
                except:
                    pass
        positive_points = generate_random_positive_points_HC(args, buffer_img, cur_box, draw_path, i_box)

        if not len(positive_points) == 0:
            prompt_points_dict[cur_box[0].item()].extend(positive_points)
        if not len(negative_points) == 0:
            prompt_points_dict[-1].extend(negative_points)
  
        positive_labels = torch.tensor([1 for _ in range(len(positive_points))], device=args.device) 
        
        negative_points = torch.tensor(negative_points, device=args.device).float()
        negative_labels = torch.tensor([0 for _ in range(len(negative_points))], device=args.device)  

        prompt_points = torch.cat((positive_points, negative_points), dim=0)
        prompt_labels = torch.cat((positive_labels, negative_labels), dim=0)
        
        if len(prompt_points) != 0:
            prompt_points = args.predictor.transform.apply_coords_torch(prompt_points, image.shape[:2])

            masks_in_buffer_img, _, _ = args.predictor.predict(
                point_coords=prompt_points,  
                point_labels=prompt_labels,
                box=None,
                multimask_output=False,
            )
            for _ in masks_in_buffer_img:
                classID = [cur_box[0]]
                classID.extend(buffer_bbox)
                mask_box.append(classID)
            masks_in_buffer_img = torch.tensor(masks_in_buffer_img.squeeze(0))
            masks_in_curr_img.append(masks_in_buffer_img)
    return masks_in_curr_img, mask_box




sam = sam_model_registry[args.model_type](checkpoint=args.weights).to(device)
predictor = SamPredictor(sam)
if len(boxes)!= 0:    
    background_area = torch.ones((original_height, original_width), dtype=bool, device=args.device)  # H W
    for box in boxes:
        _, x_min, y_min, x_max, y_max = box
        background_area[y_min:y_max, x_min:x_max] = False
    masks, mask_box = multi_point_prompt(args, boxes, original_img, background_area, draw_path)