import torch
from ml_classification.dataset.LibriDataset import LibriDataset
from ml_classification.dataset.preprocess import same_seeds, preprocess_data
from torch.utils.data import DataLoader

from ml_classification.model.deepLearning.LSTMClassifier import LSTMClassifier
from ml_classification.model.deepLearning.train import train
from ml_classification.model.deepLearning.test import test
from torch.utils.tensorboard import SummaryWriter
import datetime

if __name__ == "__main__":
    # 超参数
    concat_nframes = 21
    train_ratio = 0.95
    seed = 1213
    batch_size = 512
    num_epoch = 15
    learning_rate = 1e-3
    model_path = '../model_save/LSTM.ckpt'
    input_dim = 39 * concat_nframes
    hidden_layers = 6
    hidden_dim = 512

    same_seeds(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'DEVICE: {device}')

    writer = SummaryWriter()  # Tensorboard 画图，结果存储在 ./runs 中


    # 创建MLPClassifier实例
    LSTMClassifier = LSTMClassifier(input_dim, 41, hidden_layers, hidden_dim).to(device)

    # 加载数据
    train_X, train_y = preprocess_data(split='train', feat_dir='../data/libriphone/feat', phone_path='../data/libriphone', concat_nframes=concat_nframes, train_ratio=train_ratio, random_seed=seed)
    val_X, val_y = preprocess_data(split='val', feat_dir='../data/libriphone/feat', phone_path='../data/libriphone', concat_nframes=concat_nframes, train_ratio=train_ratio, random_seed=seed)

    # 获取数据集和数据加载器
    train_set = LibriDataset(train_X, train_y)
    val_set = LibriDataset(val_X, val_y)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # 训练模型
    train(LSTMClassifier,train_loader, val_loader, num_epoch, learning_rate, model_path, device, writer)

    # 预测
    # load data
    test_X = preprocess_data(split='test', feat_dir='./data/libriphone/feat', phone_path='./data/libriphone',concat_nframes=concat_nframes)
    test_set = LibriDataset(test_X, None)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # 获取当前时间作为时间戳
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 构建带时间戳的文件名
    output_file = f'../result/LSTMprediction_{timestamp}.csv'
    # 使用带时间戳的文件名调用函数
    test(LSTMClassifier, test_loader, device, output_file)