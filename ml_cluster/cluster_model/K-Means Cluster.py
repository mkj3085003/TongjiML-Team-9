import sys
import numpy as np
import pandas as pd
import random as rd
import pandas.api.types as ptypes
from matplotlib import pyplot as plt

class myKMeans(object):

    def __init__(self, df, K, x_label, y_label):
        self.data = df.values
        self.x_label = x_label
        self.y_label =  y_label
        self.K = K                      # num clusters
        self.m = self.data.shape[0]     # num training examples
        self.n = self.data.shape[1]     # num of features
        self.result = {}
        self.centroids = np.array([]).reshape(self.n, 0)

    #初始化随机聚类中心
    def init_random_centroids(self):
        temp_centroids = np.array([]).reshape(self.n, 0)
        for i in range(self.K):
            rand = rd.randint(0, self.m-1)
            temp_centroids = np.c_[temp_centroids, self.data[rand]]

        return temp_centroids


    def fit_model(self, num_iter):
        # 初始化随机聚类中心
        self.centroids = self.init_random_centroids()
        # 开始迭代更新聚类中心，计算并更新欧氏距离
        for i in range(num_iter):
            # 首先计算欧氏距离并存储到数组中
            EucDist = np.array([]).reshape(self.m, 0)
            for k in range(self.K):
                dist = np.sum((self.data - self.centroids[:,k])**2, axis=1)
                EucDist = np.c_[EucDist, dist]
            # 取最小距离
            min_dist = np.argmin(EucDist, axis=1) + 1

            # 开始迭代
            soln_temp = {} # 临时字典，存储每次迭代的解决方案

            for k in range(self.K):
                soln_temp[k+1] = np.array([]).reshape(self.n, 0)

            for i in range(self.m):
                # 根据聚类索引重新分组数据点
                soln_temp[min_dist[i]] = np.c_[soln_temp[min_dist[i]], self.data[i]]

            for k in range(self.K):
                soln_temp[k+1] = soln_temp[k+1].T

            # 更新聚类中心为每个聚类的新均值
            for k in range(self.K):
                self.centroids[:,k] = np.mean(soln_temp[k+1], axis=0)

            self.result = soln_temp


    def plot_kmeans(self):
        # create arrays for colors and labels based on specified K
        colors = ["#"+''.join([rd.choice('0123456789ABCDEF') for j in range(6)]) \
                    for i in range(self.K)]
        labels = ['cluster_' + str(i+1) for i in range(self.K)]

        fig1 = plt.figure(figsize=(5,5))
        ax1 = plt.subplot(111)
        # plot each cluster
        for k in range(self.K):
                ax1.scatter(self.result[k+1][:,0], self.result[k+1][:,1],
                                        c = colors[k], label = labels[k])
        # plot centroids
        ax1.scatter(self.centroids[0,:], self.centroids[1,:], #alpha=.5,
                                s = 300, c = 'lime', label = 'centroids')
        plt.xlabel(self.x_label) # first column of df
        plt.ylabel(self.y_label) # second column of df
        plt.title('Plot of K Means Clustering Algorithm')
        plt.legend()

        return plt.show(block=True)


    def predict(self):
        """
        result:每个中心点的最小欧氏距离。
        centroids.T:迭代n次后的K个中心点。
        """
        return self.result, self.centroids.T


    def plot_elbow(self):
        """
        Elbow Method:
        The elbow method will help us determine the optimal value for K.
        Steps:
        1) Use a range of K values to test which is optimal
        2) For each K value, calculate Within-Cluster-Sum-of-Squares (WCSS)
        3) Plot Num Clusters (K) x WCSS

        Returns
        -------
        plot
            elbow plot - k values vs wcss values to find optimal K value.
        """

        wcss_vals = np.array([])
        for k_val in range(1, self.K):
            results, centroids = self.predict()
            wcss=0
            for k in range(k_val):
                wcss += np.sum((results[k+1] - centroids[k,:])**2)
            wcss_vals = np.append(wcss_vals, wcss)
        # Plot K values vs WCSS values
        K_vals = np.arange(1, self.K)
        plt.plot(K_vals, wcss_vals)
        plt.xlabel('K Values')
        plt.ylabel('WCSS')
        plt.title('Elbow Method')

        return plt.show(block=True)