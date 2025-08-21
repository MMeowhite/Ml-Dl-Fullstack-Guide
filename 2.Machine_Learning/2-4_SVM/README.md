pytorch实现版本（全部代码）

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MySVM_PyTorch:
    def __init__(self, C=1.0, tol=1e-3, max_iter=1000, kernel='linear', 
                 method='SMO', sigma=1.0, use_sgd=False, eta=0.001):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.kernel_type = kernel
        self.method = method
        self.sigma = sigma
        self.use_sgd = use_sgd
        self.eta = eta

    def _kernel(self, X1, X2):
        if self.kernel_type == 'linear':
            return X1 @ X2.T
        elif self.kernel_type == 'rbf':
            X1_sq = (X1 ** 2).sum(dim=1).unsqueeze(1)
            X2_sq = (X2 ** 2).sum(dim=1).unsqueeze(0)
            K = X1_sq + X2_sq - 2 * (X1 @ X2.T)
            return torch.exp(-K / (2 * self.sigma ** 2))
        else:
            raise ValueError("Unknown kernel type")

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        self.X = X
        self.y = y
        n_samples, n_features = X.shape

        if self.method == 'GD':
            # 线性SVM权重和偏置
            self.w = nn.Parameter(torch.zeros(n_features))
            self.b = nn.Parameter(torch.zeros(1))
            optimizer = optim.SGD([self.w, self.b], lr=self.eta)
            
            for it in range(self.max_iter):
                if self.use_sgd:
                    indices = torch.randperm(n_samples)
                else:
                    indices = torch.arange(n_samples)
                
                for i in indices:
                    optimizer.zero_grad()
                    xi, yi = X[i], y[i]
                    margin = yi * (xi @ self.w + self.b)
                    loss = 0.5 * torch.sum(self.w ** 2) # 正则化项
                    if margin < 1:
                        loss += self.C * (1 - margin) # 铰链损失项
                    loss.backward()
                    optimizer.step()
        
        elif self.method == 'SMO':
            # alpha参数初始化
            self.alpha = torch.zeros(n_samples, dtype=torch.float32)
            self.b = torch.tensor(0.0, dtype=torch.float32)
            K = self._kernel(X, X)
            
            iters = 0
            while iters < self.max_iter:
                alpha_changed = 0
                for i in range(n_samples):
                    Ei = torch.sum(self.alpha * y * K[:, i]) + self.b - y[i]
                    if (y[i]*Ei < -self.tol and self.alpha[i] < self.C) or \
                       (y[i]*Ei > self.tol and self.alpha[i] > 0):
                        # 选择j
                        j = i
                        while j == i:
                            j = torch.randint(0, n_samples, (1,)).item()
                        Ej = torch.sum(self.alpha * y * K[:, j]) + self.b - y[j]
                        
                        alpha_i_old = self.alpha[i].clone()
                        alpha_j_old = self.alpha[j].clone()
                        
                        if y[i] != y[j]:
                            L = max(0.0, self.alpha[j] - self.alpha[i])
                            H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                        else:
                            L = max(0.0, self.alpha[i] + self.alpha[j] - self.C)
                            H = min(self.C, self.alpha[i] + self.alpha[j])
                        if L == H:
                            continue
                        
                        eta = K[i, i] + K[j, j] - 2 * K[i, j]
                        if eta <= 0:
                            continue
                        
                        self.alpha[j] += y[j] * (Ei - Ej) / eta
                        self.alpha[j] = torch.clamp(self.alpha[j], L, H)
                        if torch.abs(self.alpha[j] - alpha_j_old) < 1e-5:
                            continue
                        self.alpha[i] += y[i]*y[j]*(alpha_j_old - self.alpha[j])
                        
                        b1 = self.b - Ei - y[i]*(self.alpha[i]-alpha_i_old)*K[i,i] - \
                            y[j]*(self.alpha[j]-alpha_j_old)*K[i,j]
                        b2 = self.b - Ej - y[i]*(self.alpha[i]-alpha_i_old)*K[i,j] - \
                            y[j]*(self.alpha[j]-alpha_j_old)*K[j,j]
                        if 0 < self.alpha[i] < self.C:
                            self.b = b1
                        elif 0 < self.alpha[j] < self.C:
                            self.b = b2
                        else:
                            self.b = (b1 + b2)/2
                        
                        alpha_changed += 1
                if alpha_changed == 0:
                    iters += 1
                else:
                    iters = 0
        else:
            raise ValueError("Unsupported method")
        return self

    def predict_one(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        if self.method == 'GD':
            return (x @ self.w + self.b).item()
        else:
            result = torch.sum(self.alpha * self.y * self._kernel(self.X, x.unsqueeze(0)).squeeze()) + self.b
            return result.item()

    def predict(self, X):
        return np.sign(np.array([self.predict_one(x) for x in X]))

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

```