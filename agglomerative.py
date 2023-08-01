import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import random
from copy import deepcopy


def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


def clusters_distance(cluster1, cluster2):
    return max([euclidean_distance(point1, point2) for point1 in cluster1 for point2 in cluster2])


def clusters_distance_2(cluster1, cluster2):

    cluster1_center = np.average(cluster1, axis=0)
    cluster2_center = np.average(cluster2, axis=0)
    return euclidean_distance(cluster1_center, cluster2_center)


class AgglomerativeClustering:

    def __init__(self, k=2, initial_k=25):
        self.k = k
        self.initial_k = initial_k

    def initial_clusters(self, points):
        groups = {}
        d = int(256 / (self.initial_k))
        for i in range(self.initial_k):
            j = i * d
            groups[(j, j, j)] = []
        for i, p in enumerate(points):
            if i % 100000 == 0:
                print('processing pixel:', i)
            go = min(groups.keys(), key=lambda c: euclidean_distance(p, c))
            groups[go].append(p)
        return [g for g in groups.values() if len(g) > 0]

    def fit(self, points):

        self.clusters_list = self.initial_clusters(points)
        while len(self.clusters_list) > self.k:
            cluster1, cluster2 = min([(c1, c2) for i, c1 in enumerate(self.clusters_list) for c2 in self.clusters_list[:i]],
                                     key=lambda c: clusters_distance_2(c[0], c[1]))

            self.clusters_list = [c for c in self.clusters_list if np.all(np.array(
                c) != np.array(cluster1)) and np.all(np.array(c) != np.array(cluster2))]

            merged_cluster = cluster1 + cluster2
            self.clusters_list.append(merged_cluster)

        self.cluster = {}
        for cl_num, cl in enumerate(self.clusters_list):
            for point in cl:
                self.cluster[tuple(point)] = cl_num

        self.centers = {}
        for cl_num, cl in enumerate(self.clusters_list):
            self.centers[cl_num] = np.average(cl, axis=0)

    def predict_cluster(self, point):

        return self.cluster[tuple(point)]

    def predict_center(self, point):

        point_cluster_num = self.predict_cluster(point)
        center = self.centers[point_cluster_num]
        return center
