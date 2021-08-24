import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


def feature_tsne(feat_dir):
    feat = torch.load(feat_dir)
    tsne = TSNE(n_components=2).fit_transform(feat)
    plt.scatter(tsne[:, 0], tsne[:, 1])
    plt.show()
    return 0


def feature_pca(feat_dir):
    feat = torch.load(feat_dir)
    pc = PCA(n_components=2).fit_transform(feat)
    plt.scatter(pc[:, 0], pc[:, 1])
    plt.show()
    return 0


feature_pca("/mnt/data/imgnet/features/trainfeat.pth")