from nltk.cluster import KMeansClusterer, euclidean_distance
import numpy as np
import os
import sys
import random as r

## recebe um path como input.
## faz parse de cada .txt na pasta de input
## le arquivos compostos por linhas no formato <object-class> <x> <y> <width> <height>
## retorna anchors

def intersection(rect1, rect2):
    dx = min(rect1[0], rect2[0]) - max(rect1[0]+rect1[2], rect2[0]+rect2[2])
    dy = min(rect1[1], rect2[1]) - max(rect1[1]+rect1[3], rect2[1]+rect2[3])
    if dx > 0 and dy > 0:
        return dx * dy
    return 0
    
def union(rect1, rect2):
    a1 = rect1[2] * rect1[3]
    a2 = rect2[2] * rect2[3]
    
    return a1 + a2 - intersection(rect1, rect2)
    
def IOU(rect1, rect2):
    intersection_area = intersection(rect1, rect2)
    union_area = union(rect1, rect2)
    
    if union_area > 0:
        return intersection_area / union_area
    return 0
    
def iou_dist_function(v1, v2):
    return 1 - IOU(v1, v2)
    

if __name__ == "__main__":

    vectors = []

    # paths = os.listdir(sys.argv[1])
    # for path in paths:
    #     f = open(sys.argv[1] + '/' + path)
    #     for line in f:
    #         vectors.append(np.array([float(l) for l in line.split()[1:]]))
    #     f.close()
    # #    if len(vectors) > 1000:
    #         break

    path_labels =  '/home/antonitsch/GitRepos/darknet/VOCdevkit/VOC2007/labels/'
    paths = open(sys.argv[1])
    for path in paths:
        path_split = path[:-5].split('/')
        f = open(path_labels + path_split[-1] + '.txt')
        for line in f:
            vectors.append(np.array([float(l) for l in line.split()[1:]]))
        f.close()
        if len(vectors) > 5:
            break 
    print(vectors)
    print('dataset loaded')
    vectors = np.array(vectors) ## train data goes here
    
    means = [vectors[r.randint(0, len(vectors)-1)] for x in range(5)] ##initial centroids go here 

    print('clustering')
    clusterer = KMeansClusterer(5, iou_dist_function, initial_means=means, avoid_empty_clusters=True)
    clusters = clusterer.cluster(vectors, True)

    anchors = clusters.means()

    print(anchors)