from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression as lr

from ml_regression.preprocess.preprocess import DataPreprocessing
from model.linearRegression import LinearRegression


def main():
    data_path = "dataset/WorldHappiness_Corruption_2015_2020.csv"
    data_processor = DataPreprocessing(data_path)

    df = data_processor.load_data()
    df.info()
    X_train, X_test, y_train, y_test = data_processor.preprocess_data(df)

    # 在这里可以继续进行模型训练和其他操作
    linear = LinearRegression(X_train.values, y_train.values)
    theta1 = linear.train(0.5, 5000)
    # theta2 = linear.fit()
    plt.plot(theta1[1])
    plt.show()
    y_test_pred = linear.predict(X_test.values)
    print('Testing accuracy on selected features: %.3f' % r2_score(y_test.values, y_test_pred))

    # model = lr()
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    # print(r2_score(y_test, y_pred))


if __name__ == "__main__":
    main()
