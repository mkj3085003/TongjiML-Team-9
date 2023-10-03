from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression as lr

from preprocess.preprocess import DataPreprocessing
from model.linearRegression import LinearRegression
from model.mlpRegression import MlpRegression
# import missingno as msno #用于缺失值分析
import numpy as np
import torch
import torch.utils.data as Data

from preprocess.featureSelection import FeatureSelectionWithANOVA
def main():
    data_path = "dataset/WorldHappiness_Corruption_2015_2020.csv"
    data_processor = DataPreprocessing(data_path)

    df = data_processor.load_data()
    # 进行数据集的分析与可视化
    # #缺失值分析
    # print(df.head())  # 打印数据框的前几行
    # print(df.isnull().sum())  # 打印每列的缺失值数量

    df.info()
    X_train, X_test, y_train, y_test = data_processor.preprocess_data(df)

    # 在这里可以继续进行模型训练和其他操作
    linear = LinearRegression(X_train.values, y_train.values)
    print(X_train.values.shape, y_train.values.shape)
    theta1 = linear.train(0.5, 5000)
    # theta2 = linear.fit()
    # print(theta2)
    # plt.plot(theta1[1])
    # plt.show()
    y_test_pred = linear.predict(X_test.values)
    print('Testing accuracy on selected features: %.3f' % r2_score(y_test.values, y_test_pred))

    # model = lr()
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    # print(r2_score(y_test, y_pred))

    # MLP回归
    # 将数据集转化为张量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_xt = torch.from_numpy(X_train.values.astype(np.float32))
    train_yt = torch.from_numpy(y_train.values.astype(np.float32))
    test_xt = torch.from_numpy(X_test.values.astype(np.float32))
    test_yt = torch.from_numpy(y_test.values.astype(np.float32))

    # 将训练数据处理为数据加载器
    train_data = Data.TensorDataset(train_xt, train_yt)
    test_data = Data.TensorDataset(test_xt, test_yt)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=0)
    mlp = MlpRegression()
    # 定义不同的优化器
    optimizers = [
        torch.optim.SGD(mlp.parameters(), lr=0.01),
        torch.optim.Adam(mlp.parameters(), lr=0.01),
        torch.optim.RMSprop(mlp.parameters(), lr=0.01)
    ]
    # 训练不同优化器的模型并记录损失
    train_loss_all = mlp.train(train_loader, optimizers[0], num_epochs=500)  # 训练模型
    print(train_loss_all)



if __name__ == "__main__":
    main()
