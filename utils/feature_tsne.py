import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


def feature_tsne(feat_dir, label_dir):
    feat = torch.load(feat_dir)
    labels = torch.load(label_dir)
    tsne = TSNE(n_components=2).fit_transform(feat)
    plt.scatter(tsne[:, 0], tsne[:, 1])
    plt.scatter(tsne[labels == 19, 0], tsne[labels == 19, 1])
    #plt.scatter(tsne[labels == 359, 0], tsne[labels == 359, 1])
    #plt.scatter(tsne[labels == 360, 0], tsne[labels == 360, 1])
    #plt.scatter(tsne[labels == 361, 0], tsne[labels == 361, 1])
    #plt.scatter(tsne[labels == 362, 0], tsne[labels == 362, 1])
    #plt.scatter(tsne[labels == 363, 0], tsne[labels == 363, 1])
    #plt.scatter(tsne[labels == 364, 0], tsne[labels == 364, 1])
    #plt.scatter(tsne[labels == 365, 0], tsne[labels == 365, 1])
    #plt.scatter(tsne[labels == 366, 0], tsne[labels == 366, 1])
    #plt.scatter(tsne[labels == 367, 0], tsne[labels == 367, 1])
    #plt.scatter(tsne[labels == 427, 0], tsne[labels == 427, 1])
    #plt.scatter(tsne[labels == 499, 0], tsne[labels == 499, 1])
    #plt.scatter(tsne[labels == 500, 0], tsne[labels == 500, 1])
    #plt.scatter(tsne[labels == 501, 0], tsne[labels == 501, 1])
    plt.scatter(tsne[labels == 360, 0], tsne[labels == 360, 1])
    plt.scatter(tsne[labels == 367, 0], tsne[labels == 367, 1])
    plt.scatter(tsne[labels == 359, 0], tsne[labels == 359, 1])
    plt.scatter(tsne[labels == 365, 0], tsne[labels == 365, 1])
    #plt.scatter(tsne[labels == 85, 0], tsne[labels == 85, 1])
    #plt.scatter(tsne[labels == 106, 0], tsne[labels == 106, 1])
    #plt.scatter(tsne[labels == 107, 0], tsne[labels == 107, 1])
    #plt.scatter(tsne[labels == 108, 0], tsne[labels == 108, 1])
    plt.legend(["All", "WT", "0", ".6N", "6N", "60n"])
    #plt.legend(["All", "WT", "6N", "0", "2000N", "200N", "20N", "2N", "60N", "0.2N", "0.6N"])
    #plt.legend(["All", "Rooster", "Beach", "Dog", "Dog2"])
    plt.show()
    return 0


def feature_pca(feat_dir, label_dir):
    feat = torch.load(feat_dir)
    labels = torch.load(label_dir)
    tsne = PCA(n_components=2).fit_transform(feat)
    plt.scatter(tsne[:, 0], tsne[:, 1])
    plt.scatter(tsne[labels == 19, 0], tsne[labels == 19, 1])
    plt.scatter(tsne[labels == 427, 0], tsne[labels == 427, 1])
    plt.scatter(tsne[labels == 499, 0], tsne[labels == 499, 1])
    plt.scatter(tsne[labels == 500, 0], tsne[labels == 500, 1])
    plt.scatter(tsne[labels == 501, 0], tsne[labels == 501, 1])
    #plt.scatter(tsne[labels == 85, 0], tsne[labels == 85, 1])
    #plt.scatter(tsne[labels == 106, 0], tsne[labels == 106, 1])
    #plt.scatter(tsne[labels == 107, 0], tsne[labels == 107, 1])
    #plt.scatter(tsne[labels == 108, 0], tsne[labels == 108, 1])
    plt.legend(["All", "WT", "Large", "Unusual", "Immature", "Small"])
    #plt.legend(["All", "Rooster", "Beach", "Dog", "Dog2"])
    plt.show()
    return 0


feature_tsne("/mnt/data/feature_extraction/features/trainfeat.pth", "/mnt/data/feature_extraction/features/trainlabels.pth")
