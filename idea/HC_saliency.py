import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

def Quantize(img3f,ratio=0.95,colorNums=(12,12,12)):
    clrTmp = clrTmp = [colorNum-0.0001 for colorNum in colorNums]  
    w = [colorNums[1] * colorNums[2], colorNums[2], 1]  
    img3f_0,img3f_1,img3f_2 = cv2.split(img3f)
    idx_img3f_0 = (img3f_0 * clrTmp[0] ).astype(np.int32)* w[0]
    idx_img3f_1 = (img3f_1 * clrTmp[1] ).astype(np.int32)* w[1]
    idx_img3f_2 = (img3f_2 * clrTmp[2] ).astype(np.int32)* w[2]
    idx1i = idx_img3f_0 + idx_img3f_1 + idx_img3f_2
    bincount_pallet = np.bincount(idx1i.reshape(1,-1)[0])  
    sort_pallet = np.sort(bincount_pallet)  
    argsort_pallet = np.argsort(bincount_pallet)  
    numpy_pallet = np.vstack((sort_pallet, argsort_pallet))
    numpy_pallet = numpy_pallet[:, np.nonzero(sort_pallet)] 
    num = np.swapaxes(numpy_pallet, 0, 1)[0] 
    len_num = maxNum = len(num[0]) 
    height,width = img3f.shape[:2]   
    maxDropNum = int(np.round(height * width * (1 - ratio))) 
    sum_pallet = np.add.accumulate(num[0])  
    arg_sum_pallett = np.argwhere(sum_pallet >= maxDropNum)[0][0] 
    minNum = maxNum - arg_sum_pallett
    num_values = num[1][::-1] 
    minNum = 256 if minNum > 256 else minNum
    if minNum <= 10:
        minNum = 10 if len(num) > 10 else len(num)    
    color3i_init0 = (num_values / w[0]).astype(np.int32)
    color3i_init1 = (num_values % w[0]/w[1]).astype(np.int32)
    color3i_init2 = (num_values % w[1]).astype(np.int32)
    color3i = np.array([color3i_init0,color3i_init1,color3i_init2]).T
    zero2maxNum = color3i[:minNum] 
    maxNum2len_Num = color3i[minNum:] 
    temp_matrix = np.zeros((len_num-minNum,minNum),dtype=np.int32)
    for i,single in enumerate(maxNum2len_Num): 
        temp_matrix[i] = np.sum(np.square(single-zero2maxNum),axis=1)
    arg_min = np.argmin(temp_matrix, axis=1) 
    replaceable_colors = num_values[arg_min] 
    pallet = dict(zip(num_values[:minNum], range(minNum)))
    for num_value,index_dist in zip(num_values[minNum:],replaceable_colors):
        pallet[num_value] = pallet[index_dist]
    
    idx1i_reshape = idx1i.copy().reshape(1,-1)[0]
    idx1i_0 = np.zeros(height * width, dtype=np.int32)
    for i, v in enumerate(idx1i_reshape):
        idx1i_0[i] = pallet[v]
        
    idx1i = idx1i_0.reshape((height,width))
    color3f = np.zeros((1, minNum, 3), np.float32)
    colorNum = np.zeros((1, minNum), np.int32)
    np.add.at(color3f[0], idx1i, img3f)  
    np.add.at(colorNum[0], idx1i, 1)  
    colorNum_reshape = colorNum.reshape(color3f.shape[1],1)
    color3f[0] /= colorNum_reshape
    return color3f.shape[1],idx1i,color3f,colorNum

def GetHC(img_float,delta=0.25):
    binN, idx1i, binColor3f, colorNums1i = Quantize(img_float)               
    binColor3f = cv2.cvtColor(binColor3f, cv2.COLOR_BGR2Lab)               
    weight1f = np.zeros(colorNums1i.shape, np.float32)
    cv2.normalize(colorNums1i.astype(np.float32), weight1f, 1, 0, cv2.NORM_L1) 

    binColor3f_reshape = binColor3f.reshape(-1, 3)[:binN]
    similar_dist = squareform(pdist(binColor3f_reshape))
    similar_dist_sort = np.sort(similar_dist)
    similar_dist_argsort = np.argsort(similar_dist)

    weight1f = np.tile(weight1f, (binN, 1))
    color_weight_dist = np.sum(np.multiply(weight1f, similar_dist), axis=1)   

    colorSal = np.zeros((1, binN), np.float64)
    if colorSal.shape[1] < 2:
        return
    tmpNum = int(np.round(binN * delta))                                    
    n = tmpNum if tmpNum > 2 else 2

    similar_nVal = similar_dist_sort[:, :n]
    totalDist_similar = np.sum(similar_nVal, axis=1)
    every_Dist = np.tile(totalDist_similar[:, np.newaxis], (1, n)) - similar_nVal

    idx = similar_dist_argsort[:, :n]
    val_n = np.take(color_weight_dist,idx)           

    valCrnt = np.sum(val_n[:, :n] * every_Dist, axis=1)
    newSal_img = valCrnt / (totalDist_similar * n)
    cv2.normalize(newSal_img, newSal_img, 0, 1, cv2.NORM_MINMAX)            
    salHC_img = np.take(newSal_img,idx1i)
    cv2.GaussianBlur(salHC_img, (3, 3), 0, salHC_img)
    cv2.normalize(salHC_img, salHC_img, 0, 1, cv2.NORM_MINMAX)
    return salHC_img

def plt_shows(titles_l, imgs_l):
    for t,im in zip(titles_l,imgs_l):
        plt.figure()
        plt.title(t)
        plt.imshow(im,cmap='gray')
    plt.show()

def get_img3_float(img3_int):
    img3_float = img3_int.astype(np.float32)  
    img3_float = img3_float / 255.0 
    return img3_float


if __name__ == "__main__":
    input_folder = "./data"
    img_list = os.listdir(input_folder)
    for img_name in img_list:
        img_name = os.path.join(input_folder, img_name)

        img3f = get_img3_float(img_name)
        sal1 = GetHC(img3f)
        save_dir = r"result"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        BGR = cv2.cvtColor((sal1*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        save_name = os.path.join(save_dir, f"{os.path.split(img_name)[-1]}_HC.jpg")
        cv2.imwrite(save_name, BGR)