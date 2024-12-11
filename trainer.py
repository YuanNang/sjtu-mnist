import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import logging

class LogisticRegressionTrainer:
    def __init__(self, output_dir="output", lr=0.01, max_iters=5000, patience=10):
        """
        初始化训练器类。
        :param output_dir: 保存输出结果的目录
        :param lr: 学习率
        :param max_iters: 最大训练迭代次数
        :param patience: 验证集损失未改善的最大容忍次数
        """
        self.output_dir = output_dir
        self.lr = lr
        self.max_iters = max_iters
        self.patience = patience
        os.makedirs(self.output_dir, exist_ok=True)  # 创建输出目录（如果不存在）
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def save_classification_report(self, y_test, y_pred, comb):
        """
        保存分类报告到 CSV 文件。
        :param y_test: 测试集真实标签
        :param y_pred: 测试集预测标签
        :param comb: 数字组合（分类任务的类别）
        """
        report = classification_report(y_test, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        df_report.to_csv(f'{self.output_dir}/classification_report_{comb[0]}_{comb[1]}.csv')

    def plot_roc_curve(self, y_test, y_prob, comb):
        """
        绘制并保存 ROC 曲线。
        :param y_test: 测试集真实标签
        :param y_prob: 测试集预测概率
        :param comb: 数字组合（分类任务的类别）
        """
        fpr, tpr, _ = roc_curve(y_test, y_prob)  # 计算 ROC 曲线
        roc_auc = auc(fpr, tpr)  # 计算 AUC 值
        plt.figure(figsize=(10, 5))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)  # 绘制基准线
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {comb[0]} vs {comb[1]}')
        plt.legend(loc="lower right")
        plt.grid()
        plt.savefig(f'{self.output_dir}/roc_curve_{comb[0]}_{comb[1]}.png')  # 保存图片
        plt.close()

    def plot_confusion_matrix(self, y_test, y_pred, comb):
        """
        绘制并保存混淆矩阵。
        :param y_test: 测试集真实标签
        :param y_pred: 测试集预测标签
        :param comb: 数字组合（分类任务的类别）
        """
        cm = confusion_matrix(y_test, y_pred)  # 计算混淆矩阵
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[comb[0], comb[1]])
        disp.plot(cmap=plt.cm.Blues)  # 使用蓝色配色显示
        plt.title(f'Confusion Matrix for {comb[0]} vs {comb[1]}')
        plt.savefig(f'{self.output_dir}/confusion_matrix_{comb[0]}_{comb[1]}.png')  # 保存图片
        plt.close()

    def train_and_save_results(self, X, y, comb, lr_model):
        """
        训练模型并保存结果。
        :param X: 输入特征数据
        :param y: 标签数据
        :param comb: 数字组合（分类任务的类别）
        :param lr_model: 逻辑回归模型实例
        """
        # 提取当前数字组合的数据
        mask = (y == comb[0]) | (y == comb[1])
        X_subset, y_subset = X[mask], y[mask]

        original_labels = comb
        y_subset = np.where(y_subset == comb[0], 0, 1)  # 将标签转换为二元标签

        # 将数据划分为训练集、验证集和测试集
        X_train, X_temp, y_train, y_temp = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # 训练模型
        lr_model.fit(X_train, y_train, X_val, y_val, comb)

        # 在测试集上评估模型性能
        y_pred = lr_model.predict(X_test)
        self.save_classification_report(y_test, y_pred, comb)
        y_prob = lr_model.predict_proba(X_test)
        self.plot_roc_curve(y_test, y_prob, comb)
        self.plot_confusion_matrix(y_test, y_pred, comb)

        # 可视化错误预测结果
        incorrect_indices = np.where(y_pred != y_test)[0]
        self.plot_incorrect_predictions(X_test, y_test, y_pred, incorrect_indices, original_labels)

        logging.info(f"Results saved for combination {comb}")

    def plot_incorrect_predictions(self, X_test, y_test, y_pred, incorrect_indices, original_labels):
        """
        可视化测试集中预测错误的样本。
        :param X_test: 测试集特征
        :param y_test: 测试集真实标签
        :param y_pred: 测试集预测标签
        :param incorrect_indices: 错误预测的样本索引
        :param original_labels: 原始数字标签
        """
        num_incorrect = len(incorrect_indices)
        num_pages = (num_incorrect // 25) + (1 if num_incorrect % 25 != 0 else 0)

        for page in range(num_pages):
            plt.figure(figsize=(10, 10))
            start_idx = page * 25
            end_idx = min(start_idx + 25, num_incorrect)
            incorrect_subset = incorrect_indices[start_idx:end_idx]

            for i, idx in enumerate(incorrect_subset):
                plt.subplot(5, 5, i + 1)
                plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')  # 绘制灰度图像

                # 显示真实和预测标签
                true_label = original_labels[0] if y_test[idx] == 0 else original_labels[1]
                pred_label = original_labels[0] if y_pred[idx] == 0 else original_labels[1]
                plt.title(f"T: {true_label} P: {pred_label}")
                plt.axis('off')  # 隐藏坐标轴

            plt.tight_layout()
            page_filename = f"{self.output_dir}/{original_labels}_incorrect_predictions_page_{page + 1}.png"
            plt.savefig(page_filename)  # 保存结果
            plt.close()
            logging.info(f"Saved page {page + 1} of incorrect predictions as {page_filename}")
