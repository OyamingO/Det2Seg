import os
from collections import Counter
from PIL import Image
from pathlib import Path
import shutil
import argparse
import numpy as np


def voting_on_masks(path1, path2, path3, path4, out_folder, filename):
    file_exist = [os.path.isfile(path1), os.path.isfile(path2), os.path.isfile(path3), os.path.isfile(path4)]
    idx = np.where(file_exist)[0] + 1 
    for i in range(len(idx)):
        if idx[i] == 1:
            img1 = Image.open(path1)
        elif idx[i] == 2:
            img2 = Image.open(path2)
        elif idx[i] == 3:
            img3 = Image.open(path3)
        elif idx[i] == 4:
            img4 = Image.open(path4)
        else:
            print("--------- error!")

    if len(idx) == 1:
        print(f"only one vote: ----------- {filename}")
        output_path = os.path.join(out_folder + "_only_one", filename)  
        shutil.copy(eval(f"path{idx[0]}"), output_path)    
        return None
    else:
        result = Image.new("L", eval(f"img{idx[0]}").size)
        for x in range(result.width):
            for y in range(result.height):
                pix_val = []
                for i in range(len(idx)):
                    pix_val.append(eval(f"img{idx[i]}").getpixel((x, y)))
                for i in range(4):
                    if i+1 not in idx:
                        pix_val.insert(i, None)
 
                pixels = [pix_val[0], pix_val[1], pix_val[2], pix_val[3]]
                vote_count = Counter(pixels)
                for i in range(4):
                    if pix_val[i] == None:
                        vote_count[pix_val[i]] = 0
         
                vote_count[pix_val[0]] += 1.5 
                if vote_count[pix_val[0]]==5.5 or vote_count[pix_val[0]]==4.5 or vote_count[pix_val[0]]==3.5:
                    win_pixel_value = pix_val[0]
                elif vote_count[pix_val[0]] == 2.5 and vote_count[pix_val[1]] <= 2 and \
                    vote_count[pix_val[2]] <= 2   and vote_count[pix_val[3]] <= 2:
                    win_pixel_value = pix_val[0]

                elif vote_count[pix_val[0]] == 2.5 and vote_count[pix_val[1]] == 3:
                    win_pixel_value = pix_val[1]
                elif vote_count[pix_val[0]] == 2.5 and vote_count[pix_val[2]] == 3:
                    win_pixel_value = pix_val[2]
                elif vote_count[pix_val[0]] == 2.5 and vote_count[pix_val[3]] == 3:
                    win_pixel_value = pix_val[3]
                else:
                    print(f"ratio: {vote_count[pix_val[0]]}: {vote_count[pix_val[1]]}: \
                                {vote_count[pix_val[2]]}: {vote_count[pix_val[3]]}" )
                result.putpixel((x, y), win_pixel_value)
    return result


voting_on_masks(input_img1, input_img2, input_img3, input_img4, out_folder, filename)