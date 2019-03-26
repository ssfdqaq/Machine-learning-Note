import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

train = pd.read_csv('breast-cancer-train.csv')
test = pd.read_csv('breast-cancer-test.csv')

test_negative = test.loc[test['Type']==0][['Clump Thickness','Cell Size']]
test_positive = test.loc[test['Type']==1][['Clump Thickness','Cell Size']]

lr = LogisticRegression()
lr.fit(train[['Clump Thickness','Cell Size']],train['Type'])
print('Testing accuracy (all training samples):',lr.score(test[['Clump Thickness','Cell Size']], test['Type']))

intercept = lr.intercept_
coef = lr.coef_[0,:]
lx = np.arange(0,12)
ly = (-intercept-lx*coef[0])/coef[1]

plt.plot(lx, ly, c='blue')
plt.scatter(test_negative['Clump Thickness'], test_negative['Cell Size'], marker='o', s=200, c='red')
plt.scatter(test_positive['Clump Thickness'], test_positive['Cell Size'], marker='x', s=150, c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()