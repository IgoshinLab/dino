import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import torch


def hierarchical_clustering(feat_dir, label_dir, label_list, label_txt):
    feat = np.array(torch.load(feat_dir))
    labels = np.array(torch.load(label_dir))
    idx_binary = np.zeros_like(labels, dtype=bool)
    for label in label_list:
        idx_binary = idx_binary | (labels == label)
    X = feat[idx_binary, :]
    labelList = labels[idx_binary]
    linked = linkage(X, 'single')
    #labeltxt = [[" "] * len(labelList)]
    #for i, lbl in enumerate(label_list):
    #    labeltxt[labelList == lbl] = label_txt[i]
    #labelList = range(1, 11)

    plt.figure(figsize=(10, 7))
    dendrogram(linked,
               orientation='top',
               labels=labelList,
               distance_sort='descending',
               show_leaf_counts=True)
    plt.show()

hierarchical_clustering("/mnt/data/feature_extraction/features/trainfeat.pth", "/mnt/data/feature_extraction/features/trainlabels.pth", [360, 367, 359, 365], ["WT", "0", ".6N", "6N", "60N"])