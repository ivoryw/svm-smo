import numpy as np
import random
from sklearn import datasets

class svm:
    def __init__(self, i_max=100, kernel="linear", C=1.0, epsilon=10E-5):
        if kernel == "linear":
            self.kernel = lambda a, b: np.dot(a.T, b)
        elif kernel == "poly":
            self.kernel = lambda a, b: (a * b.T)**2
        elif kernel == "gauss":
            self.kernel = lambda a, b: np.exp(-np.abs(a - b)**2/(2*sigma**2))
        else:
            raise ValueError(kernel + " kernel not found")
        self.i_max = i_max
        self.C = C
        self.epsilon = epsilon

    def fit(self, X, Y):
        self.m = X.shape[0]
        self.X = X
        self.Y = Y
        self.alpha = np.zeros((self.m))
        self.b = 0.0
        it = 0
        while it < self.i_max:
            n_changed = 0
            for i in range(0,self.m-1):
                E_i = self.__f(X[i,:]) - Y[i]
                if (Y[i] * E_i < -self.epsilon and self.alpha[i] < self.C) or (Y[i] * E_i > self.epsilon and self.alpha[i] > 0):
                    j = i
                    while j == i:
                        j = random.randint(0,self.m-1)
                    E_j = self.__f(X[j,:]) - Y[j]
                    a_i_old = self.alpha[i]
                    a_j_old = self.alpha[j]
                    if Y[i] != Y[j]:
                        L = max(0.0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0.0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])
                    if L == H:
                        continue
                    k_ii = self.kernel(X[i,:], X[i,:])
                    k_ij = self.kernel(X[i,:], X[j,:])
                    k_jj = self.kernel(X[j,:], X[j,:])
                    eta = 2 *  k_ij - k_ii - k_jj
                    if eta >= 0:
                        continue
                    self.alpha[j] -= Y[j] * (E_i - E_j) / eta
                    if self.alpha[j] > H:
                        self.alpha[j] = H
                    elif self.alpha[j] < L:
                        self.alpha[j] = L
                    if abs(self.alpha[j] - a_j_old) < 10E-5:
                        continue
                    self.alpha[i] += Y[i] * Y[j] * (a_j_old - self.alpha[j])
                    b_1 = self.b - E_i - Y[i] * (self.alpha[i] - a_i_old) * k_ii - Y[j] * (self.alpha[j] - a_j_old) * k_ij
                    b_2 = self.b - E_j - Y[i] * (self.alpha[i] - a_i_old) * k_ij - Y[i] * (self.alpha[j] - a_j_old) * k_jj
                    if 0 < self.alpha[i] and self.alpha[i] < self.C:
                        self.b = b_1
                    elif 0 < self.alpha[j] and self.alpha[j] < self.C:
                        self.b = b_2
                    else:
                        self.b = (b_1 + b_2)/2
                    n_changed += 1
            if n_changed == 0:
                it += 1
            else:
                it = 0

    def __f(self, X):
        f = [self.alpha[i] * self.Y[i] * self.kernel(self.X[i,:], X) for i in range(0,self.m-1)]
        f = np.sum(f)
        return f

    def predict(self, X):
        if type(X) is not np.ndarray:
            X = np.array(X)
        f = np.array([self.__f(X[i,:]) for i in range(X.shape[0])])
        return np.sign(f)

iris = datasets.load_iris()
X = iris["data"][:,(2,3)]
Y = (iris["target"] == 2).astype(np.float64)
Y[Y==0] = -1
model = svm()
model.fit(X, Y)
print model.predict([[5.5,1.7]])
