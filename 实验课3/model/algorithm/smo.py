import random

import numpy as np


class SMO:
    def __init__(self, C, tolerance=1e-5, max_iteration=10, loss="L1"):
        self.C = C
        self.tolerance = tolerance
        self.max_iteration = max_iteration
        self.loss = loss

    def solve(self, X, Y):
        batch, features = X.shape
        alpha = np.zeros((batch, 1))
        E = np.zeros((batch, 1))
        b = 0
        flag_inter_over_set = True
        flag_iteration_count = 0
        flag_alpha_changed = 0
        while (flag_iteration_count < self.max_iteration) and \
                (flag_alpha_changed > 0 or flag_inter_over_set):
            flag_alpha_changed = 0
            if flag_inter_over_set:

                flag_inter_over_set = False
                for i in range(batch):
                    flag_iteration_count += 1
                    status, data = self.inner_iteration(X, Y, E, i, alpha, b, batch, loss=self.loss)
                    if status:
                        flag_alpha_changed += 1
                        b, alpha = data
            else:
                out_bounds = np.nonzero((alpha > 0) * (alpha < self.C))[0]  # 遍历不在边界0和C的alpha
                for i in out_bounds:
                    flag_iteration_count += 1
                    status, data = self.inner_iteration(X, Y, E, i, alpha, b, batch)
                    if status:
                        flag_alpha_changed += 1
                        b, alpha = data
                if not flag_alpha_changed:
                    flag_inter_over_set = True

        return alpha, b

    def inner_iteration(self, X, Y, E, i, alpha, b, batch, loss="L1"):
        assert loss in ["L1", "L2"], "Loss must be L1 or L2"
        E[i] = b + (alpha * Y).T @ X[:, i] - Y[i]

        if loss == "L1":
            cond1 = (Y[i] * E[i] < -self.tolerance and alpha[i] < self.C)
            cond2 = (Y[i] * E[i] > self.tolerance and alpha[i] > 0)
        else:
            cond1 = (Y[i] * E[i] < -self.tolerance)
            cond2 = (Y[i] * E[i] > self.tolerance and alpha[i] > 0)
        if cond1 or cond2:

            j = np.random.randint(0, batch)
            while j == i:

                j = np.random.randint(0, batch)

            E[j] = b + (alpha * Y).T @ X[:, j] - Y[j]

            previous_alpha_i = alpha[i]
            previous_alpha_j = alpha[j]

            if loss == "L1":
                eta = 2 * X[i, j] - X[i, i] - X[j, j]
            else:
                eta = 2 * X[i, j] - X[i, i] - X[j, j] - 1 / self.C
            if eta >= 0:
                return False, None

            alpha[j] = alpha[j] - (Y[j] * (E[i] - E[j])) / eta

            if loss == "L1":
                if Y[i] != Y[j]:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(self.C, self.C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[i] + alpha[j] - self.C)
                    H = min(self.C, alpha[i] + alpha[j])
                if L == H:
                    return False, None
                # 修剪
                alpha[j] = min(H, alpha[j])
                alpha[j] = max(L, alpha[j])
            else:
                alpha[j] = max(0, alpha[j])

            if np.allclose(alpha[j], previous_alpha_j):
                return False, None

            alpha[i] = alpha[i] + Y[i] * Y[j] * (previous_alpha_j - alpha[j])

            i_update = Y[i] * (alpha[i] - previous_alpha_i)
            j_update = Y[j] * (alpha[j] - previous_alpha_j)
            b1 = b - E[i] - i_update * X[i, i] - j_update * X[i, j]
            b2 = b - E[j] - i_update * X[i, j] - j_update * X[j, j]
            # 计算 b
            if loss == "L1":
                if 0 < alpha[i] < self.C:
                    b = b1
                elif 0 < alpha[j] < self.C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
            else:
                if 0 < alpha[i]:
                    b = b1
                elif 0 < alpha[j]:
                    b = b2
                else:
                    b = (b1 + b2) / 2
        return True, (b, alpha)
