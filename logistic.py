import numpy as np
class Logistic():
    def __init__(self,train_data,train_goal,theta):
        self.train_data = train_data
        self.train_goal = train_goal
        self.theta = theta
    def sigmoid(self):
        result = 1/(1 + np.exp(-1 * np.dot(self.train_data, self.theta)))
        #print(result[0])
        return  result
    def Gradient(self):
        result = np.dot(self.train_data.T, (self.train_goal - self.sigmoid().T).T)
        return result
    def train(self,d,learn_rate):
        for i in range(d):
            self.theta = self.theta + learn_rate*self.Gradient()

    """def Loss(self):
        predict = self.sigmoid()
        #print(predict[0])
        predict[self.train_goal == 0] = 1 - predict[self.train_goal == 0]
        result = -1*np.sum(np.log(predict))
        return result"""
    def predict(self, test):
        pre=np.dot(test,self.theta)
        pre1 = np.where(pre >=0, 1, pre)
        pre2 = np.where(pre < 0, 0, pre1)
        return pre2