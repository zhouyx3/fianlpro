import numpy as np
import pandas as pd
import random as rd
from sklearn import metrics


import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("/Users/zhouyx/Desktop/cs182/final pro/titanic/train.csv")

df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
drop_col = ['PassengerId', 'Cabin', 'Ticket']
PassengerId = df['PassengerId']
df_drop = df.drop(drop_col, axis=1)

df_drop.isnull().sum()
df_drop['FamilySize'] = df_drop['SibSp'] + df_drop['Parch'] + 1
df_drop['FamilySize'].value_counts()

df_drop['IsAlone'] = 1
df_drop['IsAlone'].loc[df_drop['FamilySize'] > 1] = 0
df_drop['IsAlone'].value_counts()

df_drop['Title'] = df['Name'].str.split(',', expand=True)[1].str.split('.', expand=True)[0]
df_drop['Title'].value_counts()

title_names = (df_drop['Title'].value_counts() < 10)

df_drop['Title'] = df_drop['Title'].apply(lambda x: 'Misc' if title_names.loc[x]==True else x)
df_drop['Title'].value_counts()

df_drop['AgeBin'] = pd.cut(df_drop['Age'].astype(int), 5)
df_drop['AgeBin'].value_counts()

df_drop['FareBin'] = pd.qcut(df_drop['Fare'], 4)
df_drop['FareBin'].value_counts()
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()
df_drop['Sex_Code'] = label.fit_transform(df_drop['Sex'])
df_drop['Embarked_Code'] = label.fit_transform(df_drop['Embarked'])
df_drop['Title_Code'] = label.fit_transform(df_drop['Title'])
df_drop['AgeBin_Code'] = label.fit_transform(df_drop['AgeBin'])
df_drop['FareBin_Code'] = label.fit_transform(df_drop['FareBin'])

df_drop.head()

feat_columns = ['Pclass', 'Sex_Code', 'Embarked_Code',
                'Title_Code', 'AgeBin_Code', 'FareBin_Code', 'FamilySize']
target = 'Survived'

data_X = df_drop[feat_columns]
data_X.head()

data_y = df_drop[target]
data_y.head()



from sklearn import model_selection

train_x, test_x, train_y, test_y = model_selection.train_test_split(data_X.values, data_y.values, random_state=0)


"""from logistic import Logistic


#logistic
theta = np.random.random((train_x.shape[1],1))
LR = Logistic(train_x, train_y,theta)
LR.train(d = 100000,learn_rate=0.0001)
pred=LR.predict(test_x)

#print(pred)
print(metrics.accuracy_score(pred, test_y))"""


# SVM with smo
from svmtest import SVM
train_y_svm=np.where(train_y ==0, -1, train_y)
test_y_svm=np.where(test_y ==0, -1, test_y)

#print(train_y_svm)
#svm = SVM(1000,1)

svm = SVM(1000,1.5)
#svm = SVM(1000,1,kernel='poly')
svm.train(train_x, train_y_svm)
a=svm.predict(test_x)
pred=np.where(a==-1,0,a)
print(metrics.accuracy_score(pred, test_y))

"""0.7309417040358744"""

"""from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='lbfgs')
model.fit(train_x, train_y)
y_pred = model.predict(test_x)
print(metrics.accuracy_score(y_pred, test_y))"""

"""from sklearn.svm import SVC

model = SVC(probability=True, gamma='auto')
model.fit(train_x, train_y)

y_pred = model.predict(test_x)
print(metrics.accuracy_score(y_pred, test_y))"""

