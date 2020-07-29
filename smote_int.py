import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import copy
import random
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import pickle
from functools import reduce

# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(row1.shape[1] - 1):
        distance += (row1[0,i] - row2[0,i]) ** 2
    return np.sqrt(distance)


def smote_data(data, labels, site):

    unique_sites = np.unique(site)
    # site_new = list()
    # for index, element in enumerate(site):
    #     if element == 'CamCAN':
    #         site_new.append(1)
    #     else:
    #         site_new.append(2)
    # print(site_new)

    camcan_site = np.where(site == 'CamCAN')
    ixi_site = np.where(site != 'CamCAN')
    # print(len(camcan_site), len(ixi_site))

    ixi_x, ixi_age = data[ixi_site], labels[ixi_site]
    camcan_x, camcan_age = data[camcan_site], labels[camcan_site]

    ixi_age = ixi_age.astype(float).round()
    ixi_age = ixi_age.astype(int)
    camcan_age = camcan_age.astype(float).round()
    camcan_age = camcan_age.astype(int)
    # print('IXI_max_min age', np.max(ixi_age), np.min(ixi_age))
    # print('CamCAN_max_min age', np.max(camcan_age), np.min(camcan_age))

    new_arr = np.array([])
    new_arr_y = np.array([])

    num = int(round(len(data) * 0.25)) # number of smote samples to create
    print('Number of smote samples', num)

    for i in range(0, num):
        rand_idx = np.random.choice(camcan_x.shape[0], size=1, replace=False)

        age_rand_idx = camcan_age[rand_idx]
        ixi_x_new = ixi_x[np.where(ixi_age == age_rand_idx)]
        # print(rand_idx, ixi_age[rand_idx], camcan_age[np.where(camcan_age == age_rand_idx)])

        if len(ixi_x_new) >= 1:
            X_new = np.vstack([camcan_x[rand_idx], ixi_x_new])
            nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X_new)
            distances, indices = nbrs.kneighbors(X_new)
            # closest_samp = X_new[indices[0, 1], :]
            # print(distances)
            # print(indices)
            # print(indices[0, 1])

            row1 = camcan_x[rand_idx]
            dist_all = []
            for row2 in ixi_x_new:
                row2 = row2.reshape(-1, 1).T
                dist_all.append(euclidean_distance(row1, row2))
            sorted_wrt_dist = np.argsort(dist_all)
            closest_samp = ixi_x_new[sorted_wrt_dist[0],:]
            # print('compare index', indices[0, 1], sorted_wrt_dist[0])

            # p = random.random()
            p = random.uniform(0,0.5)
            # print('p', p)
            syn = camcan_x[rand_idx] + p * (closest_samp - camcan_x[rand_idx])
            # print('syn', syn.shape)

            if new_arr.size == 0:
                new_arr = syn
                new_arr_y = age_rand_idx
            else:
                new_arr = np.vstack([new_arr, syn])
                new_arr_y = np.vstack([new_arr_y, age_rand_idx])

    # print('smoted data', new_arr.shape, new_arr_y.shape)

    X_train_features = np.vstack([data, new_arr])
    y_train = np.vstack([labels.reshape(-1, 1), new_arr_y])
    y_train = y_train.ravel()

    # print('total data', X_train_features.shape, y_train.shape)

    return X_train_features, y_train






















