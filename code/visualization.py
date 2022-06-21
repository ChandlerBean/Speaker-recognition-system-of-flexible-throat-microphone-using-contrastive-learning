import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

def visualization(X, y):
    pca = PCA(n_components=3)
    X_r = pca.fit(X).transform(X)
    lda = LinearDiscriminantAnalysis(n_components=3)
    X_r2 = lda.fit(X, y).transform(X)
    tsne = TSNE(n_components=3)
    X_r3 = tsne.fit_transform(X)

    # Percentage of variance explained for each components
    # print('explained variance ratio (first two components): %s' % str(pca.explained_variance_ratio_))

    colors = ["black", "blue", "brown", "chocolate", "cornsilk", "darkblue", "darkgreen", "darkgray", "darksalmon", "deeppink",
              "firebrick", "forestgreen", "gold", "goldenrod", "greenyellow", "hotpink", "lavender", "lightgreen", "linen", "navy"]
    alpha = 0.8 # 透明度
    lw = 1 # 标点大小

    ## PCA降维
    # fig = plt.figure(1)
    # ax3d = Axes3D(fig)
    # l = X_r.shape[0]
    # for i in range(l):
    #     ax3d.scatter(X_r[i, 0], X_r[i, 1], X_r[i, 2], color=colors[y[i]], alpha=alpha, lw=lw)
    # plt.legend(loc='best', shadow=False, scatterpoints=1)
    # plt.title('PCA of dataset')
    # plt.show()
    #
    # # LDA降维
    # fig = plt.figure(2)
    # ax3d = Axes3D(fig)
    # l = X_r2.shape[0]
    # for i in range(l):
    #     ax3d.scatter(X_r2[i, 0], X_r2[i, 1], X_r2[i, 2], color=colors[y[i]], alpha=alpha, lw=lw)
    # plt.legend(loc='best', shadow=False, scatterpoints=1)
    # plt.title('LDA of dataset')
    # plt.show()

    # TSNE降维
    fig = plt.figure(3)
    ax3d = Axes3D(fig)
    l = X_r.shape[0]
    for i in range(l):
        ax3d.scatter(X_r3[i, 0], X_r3[i, 1], X_r3[i, 2], color=colors[y[i]], alpha=alpha, lw=lw)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('TSNE of dataset')
    plt.show()