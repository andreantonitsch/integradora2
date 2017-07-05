from nltk.cluster import KMeansClusterer
import numpy as np
import os
import sys
import random as r

## recebe um path como input.
## faz parse de cada .txt na pasta de input
## le arquivos compostos por linhas no formato <object-class> <x> <y> <width> <height>
## retorna anchors

#interseccao de dois retangulos (w, h)
def intersection(rect1, rect2):
    overlap_x = min(rect1[0], rect2[0])
    overlap_y = min(rect1[1], rect2[1])
    if overlap_x > 0 and overlap_y > 0:
        return overlap_x * overlap_y
    return 0

# uniao de dois retangulos (w, h)    
def union(rect1, rect2):
    a1 = rect1[0] * rect1[1]
    a2 = rect2[0] * rect2[1]
    
    return a1 + a2 - intersection(rect1, rect2)

#intersection over union    
def IOU(rect1, rect2):
    intersection_area = intersection(rect1, rect2)
    union_area = union(rect1, rect2)
    
    if union_area > 0:
        return intersection_area / union_area
    return 0
    
def iou_dist_function(v1, v2):
    return 1 - IOU(v1, v2)


# clusteriza um vetor de retangulos (w,h)
def clusterize( data, repeats=50 ):

    clusterer = KMeansClusterer(5, iou_dist_function, repeats=repeats, avoid_empty_clusters=True)
    clusters = clusterer.cluster(vectors, True)
    #print(clusters)
    anchors = clusterer.means()

    return anchors


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

    #path_labels =  '/home/antonitsch/GitRepos/darknet/VOCdevkit/VOC2007/labels/'
    paths = open(sys.argv[1])
    for path in paths:
        f = open(path[:-1].replace('JPEGImages', 'labels').replace('.jpg','.txt'))
        for line in f:
            box = np.array([float(l) for l in line.split()[1:]])
            # append os w e h
            vectors.append(box[2:])
        f.close()

    print('dataset loaded')
    vectors = np.array(vectors) ## train data goes here
    
    anchors = clusterize(vectors)
    print(anchors)
    