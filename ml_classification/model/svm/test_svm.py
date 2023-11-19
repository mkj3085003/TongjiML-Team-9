from dataset.preprocess import preprocess_data
from dataset.preprocess import same_seeds
import time
import numpy as np

from model.svm import SVM, MultiClassSVM

if __name__ == "__main__":
    concat_nframes = 21
    train_ratio = 0.95
    seed = 1213
    same_seeds(seed)

    # 加载数据
    print('start loading')
    X, y = preprocess_data(split='train', feat_dir='./dataset/libriphone/feat',
                                       phone_path='./dataset/libriphone',
                                       concat_nframes=concat_nframes, train_ratio=train_ratio, random_seed=seed)
    # val_X, val_y = preprocess_data(split='val', feat_dir='./dataset/libriphone/feat', phone_path='./dataset/libriphone',
    #                                concat_nframes=concat_nframes, train_ratio=train_ratio, random_seed=seed)
    indices = np.random.choice(len(X), size=1000, replace=False)
    train_X = X[indices]
    train_y = y[indices]
    seed = 42
    indices = np.random.choice(len(X), size=200, replace=False)
    val_X = X[indices]
    val_y = y[indices]
    print('end loading')

    # 初始化
    print('start initiating')
    start = time.time()
    # svm = SVM(train_X, train_y, 10, 200, 0.001, 20)
    svm_classifier = MultiClassSVM(num_classes=41, sigma=10, C=200, toler=0.1, itertime=25, kernel='gaussian', balance_method='oversampling')
    end = time.time()
    print('end initiating')
    print('initiating time: ', end - start)

    # 开始SMO算法
    print('start training')
    start = time.time()
    # VecIndex = svm.SMO()
    svm_classifier.train(train_X, train_y)
    end = time.time()
    print('end training')
    print('training time: ', end - start)

    # 开始测试模型
    # Acc = svm.test(train_X, train_y, VecIndex)
    Acc = svm_classifier.test(val_X, val_y)
    
    print('Accurate: ', Acc)
