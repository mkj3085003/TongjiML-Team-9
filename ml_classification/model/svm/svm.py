#!/usr/bin/env python
# coding: utf-8

from collections import Counter
import numpy as np
import math
import torch


class SVM:
    '''初始化参数'''

    def __init__(self, train_data, train_label, sigma, C, toler, itertime, kernel='gaussian', degree=8, coef0=0.0):

        self.train_data = train_data  # 训练集数据
        self.train_label = train_label  # 训练集标记
        self.m, self.n = np.shape(train_data)  # self.m为训练集样本容量，self.n为特征数量
        self.sigma = sigma  # 高斯核分母上的超参数
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.KernalMatrix = self.CalKernalMatrix()  # 核矩阵
        self.alpha = np.zeros(self.m)  # 初始化拉格朗日向量，长度为训练集样本容量
        self.b = 0  # 初始化参数b
        self.C = C  # 惩罚参数
        self.toler = toler  # 松弛变量
        self.itertime = itertime  # 迭代次数
        # 初始化Elist，因为alpha和b初始值为0，因此E的初始值为训练集标记的值
        self.E = [float(-1 * y) for y in self.train_label]

    '''计算核矩阵'''

    def CalKernalMatrix(self):
        if self.kernel == 'linear':
            return np.dot(self.train_data, self.train_data.T)
        elif self.kernel == 'poly':
            return (np.dot(self.train_data, self.train_data.T) + self.coef0) ** self.degree
        elif self.kernel == 'sigmoid':
            return np.tanh(np.dot(self.train_data, self.train_data.T) + self.coef0)
        elif self.kernel == 'gaussian':
            # 转换为torch张量
            X = torch.from_numpy(self.train_data)

            # 计算样本之间的欧氏距离平方
            X_squared = torch.sum(X ** 2, dim=1, keepdim=True)
            distances = X_squared - 2 * torch.matmul(X, X.t()) + X_squared.t()

            # 计算高斯核矩阵
            KernalMatrix = torch.exp(-distances / (2 * self.sigma ** 2))

            return KernalMatrix.numpy()

    '''计算g(xi)'''

    def Calgxi(self, i):

        Index = [index for index, value in enumerate(self.alpha) if value != 0]
        gxi = 0
        for index in Index:
            gxi += self.alpha[index] * self.train_label[index] * \
                self.KernalMatrix[i][index]
        gxi = gxi + self.b

        return gxi

    '''判断是否符合KKT条件'''

    def isSatisfyKKT(self, i):

        # 获得alpha[i]的值
        alpha_i = self.alpha[i]

        # 计算yi * g(xi)
        gxi = self.Calgxi(i)
        yi = self.train_label[i]
        yi_gxi = yi * gxi

        # 判断是否符合KKT条件
        if -1 * self.toler < alpha_i < self.toler and yi_gxi >= 1:
            return True
        elif -1 * self.toler < alpha_i < self.C + self.toler and math.fabs(yi_gxi - 1) < self.toler:
            return True
        elif self.C - self.toler < alpha_i < self.C + self.toler and yi_gxi <= 1:
            return True
        return False

    '''SMO算法'''

    def SMO(self):

        # 迭代
        t = 0
        parameterchanged = 1
        while t < self.itertime and parameterchanged > 0:

            t += 1
            parameterchanged = 0
            '''选择两个alpha'''

            # 外层循环，选择第一个alpha
            for i in range(self.m):

                # 判断是否符合KKT条件，如果不满足，则选择该alpha为alpha1
                # 如果满足，则继续外层循环
                TorF = self.isSatisfyKKT(i)
                if TorF == False:
                    alpha1 = self.alpha[i]

                    # 从Earray得到alpha1对应的E1
                    E1 = self.E[i]

                    # 复制一个EMatrix，并令E1的位置为nan
                    # 这样在接下来找最大值和最小值时将不会考虑E1
                    # 这里需要使用copy，如果不用copy，改变EM_temp也会同时改变EMatrix
                    EM_temp = np.copy(self.E)
                    EM_temp[i] = np.nan

                    # 我们需要使|E1-E2|的值最大，由此选择E2
                    # 首先初始化maxE1_E2和E2及E2的下标j
                    maxE1_E2 = -1
                    E2 = np.nan
                    j = -1

                    # 内层循环
                    # 遍历EM_temp中的每一个Ei，得到使|E1-E2|最大的E和它的下标
                    for j_temp, Ej in enumerate(EM_temp):
                        if math.fabs(E1 - Ej) > maxE1_E2:
                            maxE1_E2 = math.fabs(E1 - Ej)
                            E2 = Ej
                            j = j_temp

                    # alpha2为E2对应的alpha
                    alpha2 = self.alpha[j]

                    '''求最优alpha1和alpha2'''

                    y1 = self.train_label[i]
                    y2 = self.train_label[j]

                    # 计算η
                    K11 = self.KernalMatrix[i][i]
                    K22 = self.KernalMatrix[j][j]
                    K12 = self.KernalMatrix[i][j]
                    eta = K11 + K22 - 2 * K12

                    # 计算alpha2_new
                    alpha2_new = alpha2 + y2 * (E1 - E2) / eta

                    # 计算上限H和下限L
                    if y1 != y2:
                        L = max([0, alpha2 - alpha1])
                        H = min([self.C, self.C + alpha2 - alpha1])
                    else:
                        L = max([0, alpha2 + alpha1 - self.C])
                        H = min([self.C, alpha2 + alpha1])

                    # 剪切alpha2_new
                    if alpha2_new > H:
                        alpha2_new = H
                    elif alpha2_new < L:
                        alpha2_new = L

                    # 得到alpha1_new
                    alpha1_new = alpha1 + y1 * y2 * (alpha2 - alpha2_new)

                    '''更新b'''
                    # 计算b1_new和b2_new
                    b1_new = -1 * E1 - y1 * K11 * \
                        (alpha1_new - alpha1) - y2 * K12 * \
                        (alpha2_new - alpha2) + self.b
                    b2_new = -1 * E2 - y1 * K12 * \
                        (alpha1_new - alpha1) - y2 * K22 * \
                        (alpha2_new - alpha2) + self.b

                    # 根据alpha1和alpha2的范围确定b_new
                    if 0 < alpha1_new < self.C and 0 < alpha2_new < self.C:
                        b_new = b1_new
                    else:
                        b_new = (b1_new + b2_new) / 2

                    '''更新E'''
                    # 首先需要更新两个alpha和b
                    self.alpha[i] = alpha1_new
                    self.alpha[j] = alpha2_new
                    self.b = b_new

                    # 计算Ei_new和Ej_new
                    E1_new = self.Calgxi(i) - y1
                    E2_new = self.Calgxi(j) - y2

                    # 更新E
                    self.E[i] = E1_new
                    self.E[j] = E2_new

                    if math.fabs(alpha2_new - alpha2) >= 0.000001:
                        parameterchanged += 1

            print('itertime: %a   parameterchanged: %a' %
                  (t, parameterchanged))

        # 最后遍历一遍alpha，大于0的下标即对应支持向量
        VecIndex = [index for index, value in enumerate(
            self.alpha) if value > 0]

        # 返回支持向量的下标，之后在预测时还需要用到
        return VecIndex

    '''计算b'''

    def OptimizeB(self):

        for j, a in enumerate(self.alpha):
            if 0 < a < self.C:
                break

        yj = self.train_label[j]
        summary = 0
        for i in range(self.alpha):
            summary += self.alpha[i] * \
                self.train_label[i] * self.KernalMatrix[i][j]

        optimiezedB = yj - summary
        self.b = optimiezedB

    '''计算单个核函数'''

    def CalSingleKernal(self, x, z):

        SingleKernal = np.exp(-1 * np.dot(x - z, x - z) /
                              (2 * np.square(self.sigma)))
        return SingleKernal

    '''单个新输入实例的预测'''

    def predict(self, x, VecIndex):

        # 决策函数计算
        # 求和项初始化
        summary = 0

        # Index中存储着不为0的alpha的下标
        for i in VecIndex:
            alphai = self.alpha[i]
            yi = self.train_label[i]
            Kernali = self.CalSingleKernal(x, self.train_data[i])

            summary += alphai * yi * Kernali

        # 最后+b
        # np.sign得到符号
        # result = np.sign(summary + self.b)
        result = summary + self.b

        return result

    '''测试模型'''

    def test(self, test_data, test_label, VecIndex):

        # 测试集实例数量
        TestNum = len(test_label)

        errorCnt = 0

        # 对每一个实例进行预测
        for i in range(TestNum):

            result = self.predict(test_data[i], VecIndex)
            if result != test_label[i]:
                errorCnt += 1

        Acc = 1 - errorCnt / TestNum

        return Acc


class MultiClassSVM:
    def __init__(self, num_classes, sigma, C, toler, itertime, kernel='gaussian', balance_method=None):
        self.num_classes = num_classes
        self.sigma = sigma
        self.C = C
        self.toler = toler
        self.itertime = itertime
        self.svm_models = []
        self.kernel = kernel
        self.balance_method = balance_method

    def balance_data(self, data, labels):
        if self.balance_method == "undersampling":
            return self.undersampling(data, labels)
        elif self.balance_method == "oversampling":
            return self.oversampling(data, labels)
        else:
            return data.numpy(), labels.numpy()

    def undersampling(self, data, labels):
        counter = Counter(labels)
        min_count = min(counter.values())
        balanced_data = []
        balanced_labels = []
        for class_label in counter.keys():
            indices = np.where(labels == class_label)[0]
            random_indices = np.random.choice(indices, min_count, replace=False)
            balanced_data.extend(data[random_indices].tolist())
            balanced_labels.extend(labels[random_indices].tolist())
        return np.array(balanced_data), np.array(balanced_labels)

    def oversampling(self, data, labels):
        counter = Counter(labels)
        max_count = max(counter.values())
        balanced_data = []
        balanced_labels = []
        for class_label in counter.keys():
            indices = np.where(labels == class_label)[0]
            random_indices = np.random.choice(indices, max_count, replace=True)
            balanced_data.extend(data[random_indices].tolist())
            balanced_labels.extend(labels[random_indices].tolist())
        return np.array(balanced_data), np.array(balanced_labels)

    def train(self, train_data, train_labels):
        # 进行类别平衡
        train_data, train_labels = self.balance_data(train_data, train_labels)
        for class_label in range(self.num_classes):
            # 将类别标签转换为二分类标签
            binary_labels = np.where(train_labels == class_label, 1, -1)
            # 创建SVM模型实例并训练
            svm_model = SVM(train_data, binary_labels, self.sigma,
                            self.C, self.toler, self.itertime, self.kernel)
            VecIndex = svm_model.SMO()
            self.svm_models.append((svm_model, VecIndex))

    def predict(self, test_data):
        confidences = []
        for i, (svm_model, VecIndex) in enumerate(self.svm_models):
            # 对每个类别的SVM模型进行预测
            confidence = svm_model.predict(test_data, VecIndex)
            confidences.append((i, confidence))
        confidences.sort(key=lambda x: x[1], reverse=True)
        return confidences[0][0]

    def test(self, test_data, test_labels, output_file='prediction.csv'):
        num_samples = test_data.shape[0]
        error_count = 0
        predictions = np.array([], dtype=np.int32)
        # 对每个测试样本进行预测
        for i in range(num_samples):
            pred = self.predict(test_data[i])
            if pred != test_labels[i]:
                error_count += 1
            predictions = np.concatenate(
                (predictions, np.array([pred], dtype=np.int32)), axis=0)
        accuracy = 1 - error_count / num_samples
        print(predictions)
        # 结果存入csv文件
        # with open(output_file, 'w') as f:
        #     f.write('Id,Class\n')
        #     for i, y in enumerate(predictions):
        #         f.write('{},{}\n'.format(i, y))

        return accuracy
