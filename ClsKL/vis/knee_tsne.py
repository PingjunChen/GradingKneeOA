# -*- coding: utf-8 -*-

import os, sys, pdb
import numpy as np
import deepdish as dd
from time import time
import matplotlib.pyplot as plt
from sklearn import manifold
from yellowbrick.text import TSNEVisualizer


# knee = dd.io.load('./data/feas1646_auto.h5')
knee = dd.io.load('./data/tsne/vgg19_feas1656_manual.h5')
X = knee["data"]
y = knee["target"]

n_samples, n_features = X.shape

tsne = manifold.TSNE(n_components=2, perplexity=20, early_exaggeration=4.0, learning_rate=1000, n_iter=1000,
            n_iter_without_progress=50, min_grad_norm=0, init='pca', method='exact', verbose=2)
Y = tsne.fit_transform(X)


plt.figure(figsize=(6, 5))
colors = ['b', 'g', 'r', 'y', 'k']
target_ids = [0, 1, 2, 3, 4]
target_labels = ["Grade 0", "Grade 1", "Grade 2", "Grade 3", "Grade 4"]
for i, c, label in zip(target_ids, colors, target_labels):
    newY = np.array([Y[ind] for ind, e in enumerate(y) if e==i])
    plt.scatter(newY[:, 0], newY[:, 1], c=c, label=label)
plt.legend()
plt.title("Features of VGG-19-Ordinal on Manual")
plt.savefig("vgg19_tsne.pdf")
# plt.tight_layout()
plt.show()
