import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import logging

class LogisticRegressionTrainer:
    def __init__(self, output_dir="output", lr=0.01, max_iters=5000, patience=10):
        self.output_dir = output_dir
        self.lr = lr
        self.max_iters = max_iters
        self.patience = patience
        os.makedirs(self.output_dir, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def save_classification_report(self, y_test, y_pred, comb):
        report = classification_report(y_test, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        df_report.to_csv(f'{self.output_dir}/classification_report_{comb[0]}_{comb[1]}.csv')

    def plot_roc_curve(self, y_test, y_prob, comb):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(10, 5))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {comb[0]} vs {comb[1]}')
        plt.legend(loc="lower right")
        plt.grid()
        plt.savefig(f'{self.output_dir}/roc_curve_{comb[0]}_{comb[1]}.png')
        plt.close()

    def plot_confusion_matrix(self, y_test, y_pred, comb):
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[comb[0], comb[1]])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix for {comb[0]} vs {comb[1]}')
        plt.savefig(f'{self.output_dir}/confusion_matrix_{comb[0]}_{comb[1]}.png')
        plt.close()


    def train_and_save_results(self, X, y, comb, lr_model):
        mask = (y == comb[0]) | (y == comb[1])
        X_subset, y_subset = X[mask], y[mask]

        original_labels = comb
        y_subset = np.where(y_subset == comb[0], 0, 1)

        X_train, X_temp, y_train, y_temp = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        lr_model.fit(X_train, y_train, X_val, y_val, comb)

        y_pred = lr_model.predict(X_test)
        self.save_classification_report(y_test, y_pred, comb)
        y_prob = lr_model.predict_proba(X_test)
        self.plot_roc_curve(y_test, y_prob, comb)
        self.plot_confusion_matrix(y_test, y_pred, comb)

        incorrect_indices = np.where(y_pred != y_test)[0]
        self.plot_incorrect_predictions(X_test, y_test, y_pred, incorrect_indices, original_labels)

        logging.info(f"Results saved for combination {comb}")


    def plot_incorrect_predictions(self, X_test, y_test, y_pred, incorrect_indices, original_labels):
        num_incorrect = len(incorrect_indices)
        num_pages = (num_incorrect // 25) + (1 if num_incorrect % 25 != 0 else 0)

        for page in range(num_pages):
            plt.figure(figsize=(10, 10))
            start_idx = page * 25
            end_idx = min(start_idx + 25, num_incorrect)
            incorrect_subset = incorrect_indices[start_idx:end_idx]

            for i, idx in enumerate(incorrect_subset):
                plt.subplot(5, 5, i + 1)
                plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')

                # 使用原始标签显示
                true_label = original_labels[0] if y_test[idx] == 0 else original_labels[1]
                pred_label = original_labels[0] if y_pred[idx] == 0 else original_labels[1]
                plt.title(f"T: {true_label} P: {pred_label}")
                plt.axis('off')

            plt.tight_layout()
            page_filename = f"{self.output_dir}/{original_labels}_incorrect_predictions_page_{page + 1}.png"
            plt.savefig(page_filename)
            plt.close()
            logging.info(f"Saved page {page + 1} of incorrect predictions as {page_filename}")
