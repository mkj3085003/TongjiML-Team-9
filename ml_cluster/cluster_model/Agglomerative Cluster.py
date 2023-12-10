import math
from measures import get_distance_measure


class myAgglomerativeHierarchicalClustering:
    def __init__(self, data, K, M):
        self.data = data
        self.N = len(data)
        self.K = K
        self.measure = get_distance_measure(M)
        self.clusters = self.init_clusters()

    def init_clusters(self):
        return {data_id: [data_point] for data_id, data_point in enumerate(self.data)}

    #找到距离最近的两个簇
    def find_closest_clusters(self):
        min_dist = math.inf
        closest_clusters = None

        clusters_ids = list(self.clusters.keys())

        for i, cluster_i in enumerate(clusters_ids[:-1]):
            for j, cluster_j in enumerate(clusters_ids[i+1:]):
                dist = self.measure(self.clusters[cluster_i], self.clusters[cluster_j])
                if dist < min_dist:
                    min_dist, closest_clusters = dist, (cluster_i, cluster_j)
        return closest_clusters

    #合并最近的两个簇，并形成新的簇集合
    def merge_and_form_new_clusters(self, ci_id, cj_id):
        new_clusters = {0: self.clusters[ci_id] + self.clusters[cj_id]}

        for cluster_id in self.clusters.keys():
            if (cluster_id == ci_id) | (cluster_id == cj_id):
                continue
            new_clusters[len(new_clusters.keys())] = self.clusters[cluster_id]
        return new_clusters

    def run_algorithm(self):
        while len(self.clusters.keys()) > self.K:
            closest_clusters = self.find_closest_clusters()
            self.clusters = self.merge_and_form_new_clusters(*closest_clusters)
            
        # 将每行数据分配到对应的簇，并将结果返回
        cluster_labels = []

        for data_point in self.data:
            matched = False  # 用于记录是否找到了匹配的点
            for cluster_id, points in self.clusters.items():
                for point in points:
                    if np.array_equal(data_point, point):
                        cluster_labels.append(cluster_id)
                        matched = True  # 找到匹配的点
                        break
                if matched:
                    break  # 如果找到了匹配的点，跳出内层循环
            # 如果没有找到匹配的点，添加一个标志值
            if not matched:
                cluster_labels.append(-1)  # 或者使用其他指示未匹配的值
                
        return cluster_labels


    def print(self):
        for id, points in self.clusters.items():
            print("Cluster: {}".format(id))
            for point in points:
                print("    {}".format(point))