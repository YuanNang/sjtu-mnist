import numpy as np
import logging

class LogisticRegressor:
    def __init__(self, lr=0.01, max_iters=1000, log_output=True, patience=5):
        """
        初始化逻辑回归模型。
        :param lr: 学习率
        :param max_iters: 最大迭代次数
        :param log_output: 是否输出日志
        :param patience: 早停的容忍次数
        """
        self.lr = lr
        self.max_iters = max_iters
        self.log_output = log_output
        self.patience = patience
        self.params = None  # 模型的权重参数
        self.bias = None    # 模型的偏置参数

    def sigmoid(self, z):
        """
        Sigmoid 激活函数。
        :param z: 输入值
        :return: 激活后的值
        """
        return 1 / (1 + np.exp(-z))

    def binary_cross_entropy(self, y_true, y_pred):
        """
        计算二元交叉熵损失。
        :param y_true: 真实标签
        :param y_pred: 预测值
        :return: 损失值
        """
        epsilon = 1e-15  # 防止数值稳定性问题
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # 将预测值限制在 (epsilon, 1 - epsilon) 范围内
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def initialize_parameters(self, num_features):
        """
        初始化模型参数。
        :param num_features: 输入特征数量
        """
        self.params = np.zeros(num_features)  # 初始化权重为零
        self.bias = 0                         # 初始化偏置为零

    def forward_propagation(self, X):
        """
        前向传播计算预测值。
        :param X: 输入特征
        :return: Sigmoid 激活后的预测值
        """
        return self.sigmoid(np.dot(X, self.params) + self.bias)

    def compute_gradients(self, X, y, y_pred):
        """
        计算梯度。
        :param X: 输入特征
        :param y: 真实标签
        :param y_pred: 预测值
        :return: 权重梯度和偏置梯度
        """
        num_samples = X.shape[0]
        grad_weights = (1 / num_samples) * np.dot(X.T, (y_pred - y))  # 权重梯度
        grad_bias = (1 / num_samples) * np.sum(y_pred - y)            # 偏置梯度
        return grad_weights, grad_bias

    def update_parameters(self, grad_weights, grad_bias):
        """
        更新模型参数。
        :param grad_weights: 权重梯度
        :param grad_bias: 偏置梯度
        """
        self.params -= self.lr * grad_weights  # 更新权重
        self.bias -= self.lr * grad_bias       # 更新偏置

    def evaluate(self, iteration, train_loss, X_eval, y_eval, best_loss, no_improvement_count, best_params, best_bias, comb):
        """
        在验证集上评估模型性能并实现早停逻辑。
        :param iteration: 当前迭代次数
        :param train_loss: 当前训练集损失
        :param X_eval: 验证集特征
        :param y_eval: 验证集标签
        :param best_loss: 当前最佳验证集损失
        :param no_improvement_count: 验证集损失未改善次数
        :param best_params: 最佳权重参数
        :param best_bias: 最佳偏置参数
        :param comb: 当前数字分类组合
        :return: 更新后的验证集损失、未改善次数、最佳参数和偏置
        """
        if (iteration + 1) % 100 == 0:  # 每 100 次迭代打印日志
            y_pred_eval = self.predict_proba(X_eval)
            eval_loss = self.binary_cross_entropy(y_eval, y_pred_eval)
            if self.log_output:
                logging.info(f'combination {comb} Iteration {iteration + 1}: Train Loss {train_loss:.4f}, eval Loss {eval_loss:.4f}')
            if eval_loss < best_loss:  # 如果验证集损失更低，更新最佳参数
                return eval_loss, 0, self.params.copy(), self.bias
            else:
                return best_loss, no_improvement_count + 1, best_params, best_bias
        return best_loss, no_improvement_count, best_params, best_bias

    def fit(self, X_train, y_train, X_eval, y_eval, comb):
        """
        训练逻辑回归模型。
        :param X_train: 训练集特征
        :param y_train: 训练集标签
        :param X_eval: 验证集特征
        :param y_eval: 验证集标签
        :param comb: 当前数字分类组合
        """
        num_samples, num_features = X_train.shape
        self.initialize_parameters(num_features)  # 初始化参数

        best_loss = float('inf')  # 最佳损失
        no_improvement_count = 0  # 验证集损失未改善次数
        best_params = self.params.copy()  # 保存最佳权重参数
        best_bias = self.bias  # 保存最佳偏置

        for iteration in range(self.max_iters):
            y_pred_train = self.forward_propagation(X_train)  # 前向传播
            train_loss = self.binary_cross_entropy(y_train, y_pred_train)  # 计算训练损失

            grad_weights, grad_bias = self.compute_gradients(X_train, y_train, y_pred_train)  # 计算梯度
            self.update_parameters(grad_weights, grad_bias)  # 更新参数

            # 验证集评估和早停检查
            best_loss, no_improvement_count, best_params, best_bias = self.evaluate(
                iteration, train_loss, X_eval, y_eval, best_loss, no_improvement_count, best_params, best_bias, comb
            )

            if no_improvement_count >= self.patience:  # 早停条件
                logging.info(f'Early stopping at iteration {iteration + 1}')
                break

        self.params = best_params  # 恢复最佳参数
        self.bias = best_bias

    def predict_proba(self, X):
        """
        预测类别的概率值。
        :param X: 输入特征
        :return: 概率值
        """
        return self.forward_propagation(X)

    def predict(self, X):
        """
        根据概率预测类别。
        :param X: 输入特征
        :return: 类别预测值（0 或 1）
        """
        probas = self.predict_proba(X)
        return (probas > 0.5).astype(int)  # 将概率值转换为 0 或 1
