from model import LogisticRegressor
from trainer import LogisticRegressionTrainer
import numpy as np

if __name__ == "__main__":
    # 读取 MNIST 数据集
    data_file_path = "data/mnist_data.npz"
    mnist = np.load(data_file_path)
    X, y = mnist['X'], mnist['y']
    
    # 分类组合
    combinations = [(4, 9), (4, 6), (0, 1), (2, 7)]

    output_dir = "output"
    # 实例化Trainer 类
    trainer = LogisticRegressionTrainer(output_dir=output_dir, lr=0.01, max_iters=5000, patience=10)
    
    # 为每一组组合训练并保存结果
    for comb in combinations:
        # 实例化 LogisticRegressor 模型
        lr_model = LogisticRegressor(lr=0.01, max_iters=5000, patience=10)
        trainer.train_and_save_results(X, y, comb, lr_model)
