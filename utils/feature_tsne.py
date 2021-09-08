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
    plt.scatter(tsne[labels == 7, 0], tsne[labels == 7, 1])
    plt.scatter(tsne[labels == 978, 0], tsne[labels == 978, 1])
    plt.scatter(tsne[labels == 267, 0], tsne[labels == 267, 1])
    plt.scatter(tsne[labels == 260, 0], tsne[labels == 260, 1])
    #plt.scatter(tsne[labels == 85, 0], tsne[labels == 85, 1])
    #plt.scatter(tsne[labels == 106, 0], tsne[labels == 106, 1])
    #plt.scatter(tsne[labels == 107, 0], tsne[labels == 107, 1])
    #plt.scatter(tsne[labels == 108, 0], tsne[labels == 108, 1])
    #plt.legend(["All", "Large", "Unusual", "Immature", "Small"])
    plt.legend(["All", "Rooster", "Beach", "Dog", "Dog2"])
    plt.show()
    return 0


def feature_pca(feat_dir, label_dir):
    feat = torch.load(feat_dir)
    labels = torch.load(label_dir)
    pc = PCA(n_components=2).fit_transform(feat)
    plt.scatter(pc[:, 0], pc[:, 1])
    plt.scatter(pc[labels == 7, 0], pc[labels == 7, 1])
    plt.scatter(pc[labels == 978, 0], pc[labels == 978, 1])
    plt.scatter(pc[labels == 267, 0], pc[labels == 267, 1])
    plt.scatter(pc[labels == 260, 0], pc[labels == 260, 1])
    #plt.scatter(pc[labels == 107, 0], pc[labels == 107, 1])
    #plt.scatter(pc[labels == 108, 0], pc[labels == 108, 1])
    #plt.legend(["All", "Large", "Unusual", "Immature", "Small"])
    plt.legend(["All", "Rooster", "Beach", "Dog", "Dog2"])
    plt.show()
    return 0


feature_pca("/mnt/data/imgnet/features/trainfeat.pth", "/mnt/data/imgnet/features/trainlabels.pth")
