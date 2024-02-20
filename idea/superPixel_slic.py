from copy import copy, deepcopy
import math
from cv2 import CHAIN_APPROX_NONE, RETR_LIST, imshow, merge, findContours, waitKey
from skimage import io, color
import numpy as np
from tqdm import trange
from tqdm import tqdm
from sklearn.cluster import KMeans
import os

class Cluster(object):
    cluster_index = 1

    def __init__(self, h, w, l=0, a=0, b=0):
        self.update(h, w, l, a, b)
        self.pixels = []
        self.no = self.cluster_index
        self.label = 0
        Cluster.cluster_index += 1

    def update(self, h, w, l, a, b):
        self.h = h
        self.w = w
        self.l = l
        self.a = a
        self.b = b

    def __str__(self):
        return "{},{}:{} {} {} ".format(self.h, self.w, self.l, self.a, self.b)

    def __repr__(self):
        return self.__str__()


class SLICProcessor(object):
    @staticmethod
    def open_image(path):
        rgb = io.imread(path)
        if path[-3:] == 'png':
            lab_arr = color.rgb2lab(rgb[:, :, 0:3])
        else:
            lab_arr = color.rgb2lab(rgb)
        return lab_arr 
    
    @staticmethod
    def image_rgb_2_arr(rgb):       
        lab_arr = color.rgb2lab(rgb)
        return lab_arr

    @staticmethod
    def save_lab_image(path, lab_arr):
        """
        Convert the array to RBG, then save the image
        :param path:
        :param lab_arr:
        :return:
        """
        rgb_arr = color.lab2rgb(lab_arr)
        rgb_arr = np.uint8(rgb_arr*255)
        io.imsave(path, rgb_arr)

    def make_cluster(self, h, w):
        h = int(h)
        w = int(w)
        return Cluster(h, w,
                       self.data[h][w][0],
                       self.data[h][w][1],
                       self.data[h][w][2])

    def __init__(self, filename=None,image=None, K=500, M=40):
        self.K = K
        self.M = M
        if filename is not None:
            self.data = self.open_image(filename)
        elif image is not None:
            self.data = self.image_rgb_2_arr(image)
        else:
            print('input loss!')
            return
        self.image_height = self.data.shape[0]
        self.image_width = self.data.shape[1]
        self.N = self.image_height * self.image_width
        self.S = int(math.sqrt(self.N / self.K))

        self.clusters = []
        self.label = {}
        self.dis = np.full((self.image_height, self.image_width), np.inf)
        

    def init_clusters(self):
        h = self.S // 2
        w = self.S // 2
        while h < self.image_height:
            while w < self.image_width:
                self.clusters.append(self.make_cluster(h, w))
                w += self.S
            w = self.S // 2
            h += self.S

    def get_gradient(self, h, w):
        if w + 1 >= self.image_width:
            w = self.image_width - 2
        if h + 1 >= self.image_height:
            h = self.image_height - 2

        gradient = self.data[h + 1][w + 1][0] - self.data[h][w][0] + \
                   self.data[h + 1][w + 1][1] - self.data[h][w][1] + \
                   self.data[h + 1][w + 1][2] - self.data[h][w][2]
        return gradient

    def move_clusters(self):
        for cluster in self.clusters:
            cluster_gradient = self.get_gradient(cluster.h, cluster.w)
            for dh in range(-1, 2):
                for dw in range(-1, 2):
                    _h = cluster.h + dh
                    _w = cluster.w + dw
                    new_gradient = self.get_gradient(_h, _w)
                    if new_gradient < cluster_gradient:
                        cluster.update(_h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])
                        cluster_gradient = new_gradient

    def assignment(self):
        for cluster in tqdm(self.clusters):
            for h in range(cluster.h - 2 * self.S, cluster.h + 2 * self.S):
                if h < 0 or h >= self.image_height: continue
                for w in range(cluster.w - 2 * self.S, cluster.w + 2 * self.S):
                    if w < 0 or w >= self.image_width: continue
                    L, A, B = self.data[h][w]
                    Dc = math.sqrt(
                        math.pow(L - cluster.l, 2) +
                        math.pow(A - cluster.a, 2) +
                        math.pow(B - cluster.b, 2))
                    Ds = math.sqrt(
                        math.pow(h - cluster.h, 2) +
                        math.pow(w - cluster.w, 2))
                    D = math.sqrt(math.pow(Dc / self.M, 2) + math.pow(Ds / self.S, 2))
                    if D < self.dis[h][w]:
                        if (h, w) not in self.label:
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        else:
                            self.label[(h, w)].pixels.remove((h, w))
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        self.dis[h][w] = D
        a = 1

    def update_cluster(self):
        for cluster in self.clusters:
            sum_h = sum_w = number = 0
            for p in cluster.pixels:
                sum_h += p[0]
                sum_w += p[1]
                number += 1
            if number > 0:
                _h = int(sum_h / number)
                _w = int(sum_w / number)
                cluster.update(_h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])

    def save_current_image(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        image_arr = np.copy(self.data)
        for cluster in self.clusters:
            for p in cluster.pixels:
                image_arr[p[0]][p[1]][0] = cluster.l
                image_arr[p[0]][p[1]][1] = cluster.a
                image_arr[p[0]][p[1]][2] = cluster.b
            image_arr[cluster.h][cluster.w][0] = 0
            image_arr[cluster.h][cluster.w][1] = 0
            image_arr[cluster.h][cluster.w][2] = 0
        output_filename = os.path.join(output_path,'corse.png')
        self.save_lab_image(output_filename, image_arr)
        mask_r = np.ones((self.image_height, self.image_width), np.uint8)*0
        for cluster in self.clusters:
            mask = np.ones((self.image_height, self.image_width), np.uint8)*0
            for x in cluster.pixels:
                mask[x[0], x[1]] = 255
            contours, _ = findContours(mask, RETR_LIST, CHAIN_APPROX_NONE)
            for contour in contours:
                for i in contour:
                    mask_r[i[0][1], i[0][0]] = 255
        for x in range(self.image_height):
            for y in range(self.image_width):
                if mask_r[x, y] == 255:
                    image_arr[x, y] = [100, 0, 0]
        output_filename = os.path.join(output_path,'fine.png')
        self.save_lab_image(output_filename, image_arr)

    def iterate_10times(self, output_path='output'):
        self.init_clusters()
        self.move_clusters()
        self.assignment()
        self.update_cluster()
    def generate_result(self, K, output_path="output"):
        clusters = deepcopy(self.clusters)
        temp_img = [[x.l, x.a, x.b] for x in clusters]
        kmeans = KMeans(n_clusters=K, random_state=3).fit(temp_img)
        for i in range(len(self.clusters)):
            self.clusters[i].label = kmeans.labels_[i]
        
        mask = np.ones((self.image_height, self.image_width))
        img = merge([mask, mask, mask])
        for cluster in self.clusters:
            for pixel in cluster.pixels:
                img[pixel[0], pixel[1]] = kmeans.cluster_centers_[cluster.label]

        for num in range(K):
            mask = np.ones((self.image_height, self.image_width), np.uint8)*0
            for cluster in self.clusters:
                if cluster.label == num:
                    for pixel in cluster.pixels:
                        mask[pixel[0], pixel[1]] = 255
            contours, _ = findContours(mask, RETR_LIST, CHAIN_APPROX_NONE)
            for contour in contours:
                for i in contour:
                   img[i[0][1], i[0][0]] = [100, 0, 0]
                   self.data[i[0][1], i[0][0]] = [100, 0, 0]
        binary_mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)

        for cluster in self.clusters:
            for pixel in cluster.pixels:
                binary_mask[pixel[0], pixel[1]] = cluster.label 
        return binary_mask, self.data


if __name__ == '__main__':
    output_path = r'./result_newslice'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    p = SLICProcessor(filename=r'/home/oyamingo/dataset/FASSD/images_84/bothFireAndSmoke_CV013000.jpg', K=500, M=4)
    p.iterate_10times(output_path)
    p.generate_result(2, output_path)
    waitKey()