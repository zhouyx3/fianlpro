import numpy as np
class SVM:
    def __init__(self, d, c ,kernel='linear'):
        self.d = d #depth
        self.C = c #
        self._kernel = kernel

    def variable(self, features, labels):
        self.m, self.n = features.shape
        self.X = features
        self.Y = labels
        self.w = 0.0
        self.alpha = np.ones(self.m)
        self.E = [self.exp(i) for i in range(self.m)]
    def kkt(self, i,toler=0.0001):
        y_g = self.g(i) * self.Y[i]
        if y_g-1<-toler and self.alpha[i] <self.C:
            return False
        elif y_g-1>toler and self.alpha[i] >0:
            return False
        else:
            return True
    def g(self, i):
        r = self.w
        for j in range(self.m):
            r += self.alpha[j] * self.Y[j] * self.kernel(self.X[i], self.X[j])
        return r
    def exp(self, i):
        return self.g(i) - self.Y[i]
    def kernel(self, x1, x2):
        if self._kernel == 'linear':
            return sum([x1[k]*x2[k] for k in range(self.n)])
        elif self._kernel == 'poly':
            return (sum([x1[k]*x2[k] for k in range(self.n)])+1)**2
        return 0

    def choose(self):
        list = [i for i in range(self.m) if 0 < self.alpha[i] < self.C]
        list_1 = [i for i in range(self.m) if i not in list]
        list_2 = [i for i in range(self.m) if self.alpha[i] < 0 or self.alpha[i] > self.C]
        list.extend(list_1)
        list_2.extend(list)
        for i in list_2:
            if not self.kkt(i):
                E1 = self.E[i]
                if E1 >= 0:
                    j = min(range(self.m), key=lambda x: self.E[x])
                else:
                    j = max(range(self.m), key=lambda x: self.E[x])
                return i, j
    def predict_data(self, data):
        r = self.w
        for i in range(self.m):
            r += self.alpha[i] * self.Y[i] * self.kernel(data, self.X[i])
        return r
    def change_w(self, i, j, alpha_i_new, alpha_j_new):
        b1 = (-self.E[i]
              - self.Y[j] * self.kernel(self.X[j], self.X[i]) * (alpha_j_new - self.alpha[j])
              - self.Y[i] * self.kernel(self.X[i], self.X[i]) * (alpha_i_new - self.alpha[i])
              + self.w)
        b2 = (-self.E[j]
              - self.Y[j] * self.kernel(self.X[j], self.X[j]) * (alpha_j_new - self.alpha[j])
              - self.Y[i] * self.kernel(self.X[i], self.X[j]) * (alpha_i_new - self.alpha[i])
              + self.w)

        if 0 < alpha_i_new < self.C:
            b_new = b1
        elif 0 < alpha_j_new < self.C:
            b_new = b2
        else:
            b_new = (b1+ b2) / 2
        self.w = b_new
        """list = [i for i in range(self.m) if 0 < self.alpha[i] < self.C]
        if len(list) == 0:
            return self.b
        b_old=self.b
        b_max=self.b
        b=0
        score=0
        for i in list:
            bi = self.Y[i]-self.predict(self.X[i])+self.b
            self.b=bi
            score1=self.score(self.X, self.Y)
            if score1>score:
                b_max=bi
        self.b=b_max"""
    def train(self, features, labels):
        self.variable(features, labels)
        for x in range(self.d):
            i, j = self.choose()
            if self.Y[i] != self.Y[j]:
                L = max(0, self.alpha[i] - self.alpha[j])
                H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
            else:
                L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                H = min(self.C, self.alpha[i] + self.alpha[j])

            alpha_i=self.alpha[i]
            alpha_j=self.alpha[j]
            e=self.kernel(self.X[i], self.X[i]) + self.kernel(self.X[j], self.X[j]) - 2*self.kernel(self.X[i], self.X[j])
            alpha_j_unc=alpha_j+self.Y[j]*(self.exp(i)-self.exp(j))/e


            alpha_j_new =alpha_j_unc
            if alpha_j_unc > H:
                alpha_j_new= H
            elif alpha_j_unc < L:
                alpha_j_new= L

            alpha_i_new=alpha_i+self.Y[i]*self.Y[j]*(alpha_j-alpha_j_new)

            self.change_w(i, j, alpha_i_new, alpha_j_new)
            self.alpha[i] = alpha_i_new
            self.alpha[j] = alpha_j_new
            self.E[i] = self.exp(i)
            self.E[j] = self.exp(j)
    def predict(self,test):
        m=len(test)
        pred = np.zeros(m)
        for i in range(m):
            if self.predict_data(test[i]) > 0:
                pred[i]=1
            else:
                pred[i]= -1
        return pred