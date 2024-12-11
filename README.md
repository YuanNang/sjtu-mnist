# Logistic Regression for MNIST Classification

本项目是上海交通大学最优化方法课程大作业，是一个简单的基于逻辑回归的二分类任务框架，针对 MNIST 数据集的不同数字组合进行分类。

## 数据集

项目使用的 MNIST 数据集以 `.npz` 格式存储，文件路径默认为 `data/mnist_data.npz`。数据集需包含以下键值：
- `X`: 图像数据，形状为 `(样本数, 28, 28)`。
- `y`: 标签数据，形状为 `(样本数,)`。

## 使用说明

### 安装依赖

本项目Python 版本为 3.12 ，然后执行以下命令安装依赖：
```bash
pip install -r requirements.txt
```
运行以下命令启动程序：
```bash
python main.py
```
### 1. 数据预处理

程序会从 MNIST 数据集中提取指定的数字组合（例如 `(4, 9)`）进行二分类，将标签转换为二值格式：
- 第一个数字标记为 `0`
- 第二个数字标记为 `1`

### 2. 模型训练与评估

运行 `main.py` 时，程序会针对以下数字组合进行分类训练（可在main.py中进行修改）：
- `(4, 9)`
- `(4, 6)`
- `(0, 1)`
- `(2, 7)`

每个组合的流程如下：
1. 将数据划分为训练集、验证集和测试集。
2. 使用逻辑回归模型进行训练，并根据验证集的性能实现早停。
3. 保存模型的分类报告、ROC 曲线和混淆矩阵。
4. 可视化分类错误的预测结果。

### 3. 输出结果

所有结果将保存在 `output` 文件夹中，包括：
- **分类报告**：每个组合的分类指标（精确率、召回率等），保存为 CSV 文件。
- **ROC 曲线**：每个组合的 Receiver Operating Characteristic 曲线及 AUC，保存为 PNG 文件。
- **混淆矩阵**：每个组合的分类混淆矩阵，保存为 PNG 文件。
- **错误预测可视化**：测试集中分类错误的样本图像，按页面分布，保存为 PNG 文件。

## 模型逻辑概述

### 1. LogisticRegressor 类
定义在 `model.py` 文件中，主要功能：
- 使用梯度下降法更新参数。
- 提供二分类逻辑回归的训练（`fit`）、预测概率（`predict_proba`）和标签（`predict`）。

### 2. LogisticRegressionTrainer 类
定义在 `trainer.py` 文件中，主要功能：
- 将数据分割为训练集、验证集和测试集。
- 训练模型。
- 保存分类报告、绘制 ROC 曲线和混淆矩阵。
- 可视化测试集中错误分类的样本。

## 参数设置

在 `main.py` 文件中可以自定义以下参数：
- **学习率（lr）**：逻辑回归的学习率，默认值为 `0.01`。
- **最大迭代次数（max_iters）**：训练的最大迭代次数，默认值为 `5000`。
- **早停容忍度（patience）**：验证集损失未改善的最大容忍次数，默认值为 `10`。

如需更改输出目录，请在 `main.py` 中调整 `output_dir` 参数。

## 示例

训练完成后，可以在 `output` 文件夹中查看以下文件（以组合 `(4, 9)` 为例）：
- 分类报告`classification_report_4_9.csv`
-  ROC 曲线`roc_curve_4_9.png`
- 混淆矩阵`confusion_matrix_4_9.png`
- 错误分类图像：`4_9_incorrect_predictions_page_1.png`, `4_9_incorrect_predictions_page_2.png` 等。

