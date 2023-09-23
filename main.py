from  preprocess.preprocess import DataPreprocessing

def main():
    data_path = "dataset/WorldHappiness_Corruption_2015_2020.csv"
    data_processor = DataPreprocessing(data_path)

    df = data_processor.load_data()
    df.info()
    X_train, X_test, y_train, y_test = data_processor.preprocess_data(df)

    # 在这里可以继续进行模型训练和其他操作

if __name__ == "__main__":
    main()