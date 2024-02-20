import os
import cv2
import torch
import argparse
import numpy as np
from pathlib import Path
from segment_anything import SamPredictor, sam_model_registry
from superPixel_slic import *


def generate_random_positive_points_SLIC(args, buffer_img, box_in_buffer, threshold_smoke):
    category_id, x1, y1, x2, y2 = box_in_buffer
    w, h = x2 - x1, y2 - y1    

    region = buffer_img[y1:y2, x1:x2]
    if w < 32 and h < 32:
        region_mask = torch.ones((region.shape[0], region.shape[1]), dtype=bool, device=args.device) 
        positive_positions = torch.tensor(get_random_XY_pts(args, region_mask, args.num_pos_pts), device=args.device).float()
        return positive_positions, buffer_img        

    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    if (h*w//2) < 500:
        p = SLICProcessor(image=region, K=h*w//3, M=40)
        p.iterate_10times()
        mask, box_img_contours = p.generate_result(2)
    else: 
        p = SLICProcessor(image=region, K=500, M=40)
        p.iterate_10times()
        mask, box_img_contours = p.generate_result(2)
    
    box_img_contours = (color.lab2rgb(box_img_contours) * 255).astype(np.uint8)  
    buffer_img[y1:y2, x1:x2] = box_img_contours
    
    center_h = h//2
    center_w = w//2
    box_size_w = w // 8 
    box_size_h = h // 8
    center_x_min = center_w - box_size_w // 2
    center_x_max = center_w + box_size_w // 2
    center_y_min = center_h - box_size_h // 2
    center_y_max = center_h + box_size_h // 2
    center_box_labels = mask[center_y_min:center_y_max, center_x_min:center_x_max]
    unique_labels, counts = np.unique(center_box_labels, return_counts=True)
    most_frequent_label = None
    if unique_labels.size > 0:
        most_frequent_label = unique_labels[np.argmax(counts)]
    cluster_fire_or_smoke = most_frequent_label
    brightness_sum = 0
    fire_or_smoke_pixel_count = 0
    
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            label = mask[row, col] 
            if label == cluster_fire_or_smoke:
                brightness = gray[row, col]  
                brightness_sum += brightness
                fire_or_smoke_pixel_count += 1
    if fire_or_smoke_pixel_count > 0:
        mean_brightness_smoke = brightness_sum / fire_or_smoke_pixel_count
    else:
        mean_brightness_smoke = 0  
    threshold_slice = fire_or_smoke_pixel_count / (h * w) 
    gray_mean = gray.mean()
    if threshold_slice < 0.55:
        if gray_mean >= 224: 
            threshold_range = threshold_smoke             
        elif gray_mean < 224: 
            threshold_range = [threshold_smoke[0]-50, threshold_smoke[0]-15]
    else:
        if mean_brightness_smoke >= 224:   
            threshold_range = threshold_smoke
        elif mean_brightness_smoke < 224:   
            threshold_range = [threshold_smoke[0]-50, threshold_smoke[0]-15]
    brightness_threshold_low = np.percentile(gray, threshold_range[0])
    brightness_threshold_high = np.percentile(gray, threshold_range[1])

    positive_positions = []
    if threshold_slice < 0.55:
        for i in range(h):
            for j in range(w):
                if gray[i, j] >= brightness_threshold_low and gray[i, j] <= brightness_threshold_high:
                    if mean_brightness_smoke >= brightness_threshold_low and mean_brightness_smoke <= brightness_threshold_high:
                        if mask[i,j] == cluster_fire_or_smoke:
                            positive_positions.append((x1 + j, y1 + i)) # W H
                    else:
                        positive_positions.append((x1 + j, y1 + i)) # W H
    else:
        for i in range(h):
            for j in range(w):
                if gray[i,j] >= brightness_threshold_low and gray[i,j] <= brightness_threshold_high:
                    if mask[i,j] ==  cluster_fire_or_smoke:
                        positive_positions.append((x1 + j, y1 + i)) # W H               
    
    random.shuffle(positive_positions) 
    positive_positions = torch.tensor(positive_positions[:args.num_pos_pts], device=args.device).float() 
    return positive_positions, buffer_img


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
        positive_points, buffer_img_contours = generate_random_positive_points_SLIC(args, buffer_img, cur_box, args.threshold_smoke)
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
