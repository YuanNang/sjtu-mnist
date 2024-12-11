import numpy as np
import logging

class LogisticRegressor:
    def __init__(self, lr=0.01, max_iters=1000, log_output=True, patience=5):
        self.lr = lr
        self.max_iters = max_iters
        self.log_output = log_output
        self.patience = patience
        self.params = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def binary_cross_entropy(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def initialize_parameters(self, num_features):
        self.params = np.zeros(num_features)
        self.bias = 0

    def forward_propagation(self, X):
        return self.sigmoid(np.dot(X, self.params) + self.bias)

    def compute_gradients(self, X, y, y_pred):
        num_samples = X.shape[0]
        grad_weights = (1 / num_samples) * np.dot(X.T, (y_pred - y))
        grad_bias = (1 / num_samples) * np.sum(y_pred - y)
        return grad_weights, grad_bias

    def update_parameters(self, grad_weights, grad_bias):
        self.params -= self.lr * grad_weights
        self.bias -= self.lr * grad_bias

    def evaluate(self, iteration, train_loss, X_eval, y_eval, best_loss, no_improvement_count, best_params, best_bias, comb):
        if (iteration + 1) % 100 == 0:
            y_pred_eval = self.predict_proba(X_eval)
            eval_loss = self.binary_cross_entropy(y_eval, y_pred_eval)
            if self.log_output:
                logging.info(f'combination {comb} Iteration {iteration + 1}: Train Loss {train_loss:.4f}, eval Loss {eval_loss:.4f}')
            if eval_loss < best_loss:
                return eval_loss, 0, self.params.copy(), self.bias
            else:
                return best_loss, no_improvement_count + 1, best_params, best_bias
        return best_loss, no_improvement_count, best_params, best_bias

    def fit(self, X_train, y_train, X_eval, y_eval, comb):
        num_samples, num_features = X_train.shape
        self.initialize_parameters(num_features)

        best_loss = float('inf')
        no_improvement_count = 0
        best_params = self.params.copy()
        best_bias = self.bias

        for iteration in range(self.max_iters):
            y_pred_train = self.forward_propagation(X_train)
            train_loss = self.binary_cross_entropy(y_train, y_pred_train)

            grad_weights, grad_bias = self.compute_gradients(X_train, y_train, y_pred_train)
            self.update_parameters(grad_weights, grad_bias)

            best_loss, no_improvement_count, best_params, best_bias = self.evaluate(
                iteration, train_loss, X_eval, y_eval, best_loss, no_improvement_count, best_params, best_bias, comb
            )

            if no_improvement_count >= self.patience:
                logging.info(f'Early stopping at iteration {iteration + 1}')
                break

        self.params = best_params
        self.bias = best_bias

    def predict_proba(self, X):
        return self.forward_propagation(X)

    def predict(self, X):
        probas = self.predict_proba(X)
        return (probas > 0.5).astype(int)
