import torch
from torch.utils.data import DataLoader

from ml_classification.dataset.LibriDataset import LibriDataset
from ml_classification.dataset.preprocess import preprocess_data, same_seeds

from torch.utils.tensorboard import SummaryWriter
import datetime

from ml_classification.model.LogisticRegressionClassifier.oneVsAllLRClassifier import OneVsAllLRClassifier

if __name__ == "__main__":
    # 超参数
    # 超参数
    concat_nframes = 21
    train_ratio = 0.5
    seed = 1213
    num_epoch = 500
    learning_rate = 1e-1
    # model_path = '../model_save/ovoLRC.ckpt'
    input_dim = 39 * concat_nframes

    same_seeds(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'DEVICE: {device}')

    writer = SummaryWriter()  # Tensorboard 画图，结果存储在 ./runs 中


    # 创建分类器实例
    ova_classifier = OneVsAllLRClassifier(input_dim=input_dim, output_dim=41, random_state=seed,num_epoch=num_epoch,learning_rate=learning_rate)

    # 加载数据
    train_X, train_y = preprocess_data(split='train', feat_dir='../data/libriphone/feat',
                                       phone_path='../data/libriphone', concat_nframes=concat_nframes,
                                       train_ratio=train_ratio, random_seed=seed)


    test_X = preprocess_data(split='test', feat_dir='../data/libriphone/feat', phone_path='../data/libriphone',concat_nframes=concat_nframes)

    # 训练模型
    print("-----------------------开始训练------------------------------")
    ova_classifier.train(train_X,  train_y, device)
    print("-----------------------训练结束------------------------------")

    # 使用 OvA 分类器进行预测

    # 保存预测结果
    test_set = LibriDataset(test_X, None)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    # 获取当前时间作为时间戳
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 构建带时间戳的文件名
    output_file = f'../result/ovaLRC_prediction_{timestamp}.csv'
    ova_classifier.predict_and_write_to_file(test_loader, device, output_file=output_file)


