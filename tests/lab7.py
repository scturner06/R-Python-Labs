# NOTES
# 1. No orthogonal polynomials in python
# 2. prediction scoring doesn't have a bias adjusted measure
# 3. no "default" dataset.
# 4. The bootstrapping function is written in a very R way.  Will redo using a
#    more python sensible approach.

# IMPORTS
import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

# READ IN DATA
autoData = pd.read_csv('../data/Auto.csv'
                       , index_col = None
                       , na_values = ['?']) # data contains ? for NA
autoData = autoData.dropna()

portData = pd.read_csv('../data/portfolio.csv',
                       index_col = 0)

# split data
train, test = train_test_split(autoData, test_size = 0.2, random_state = 1)

# regression
# note that these are approximately the R equilvent as there is no poly()
# function in Python that computes orthogonal polynomials but this serves the
# same instructive purpose
#==============================================================================
model_LR = linear_model.LinearRegression()
model_LR.fit(train[['horsepower']], train[['mpg']])
((test[['mpg']] - model_LR.predict(test[['horsepower']])) ** 2).mean()

model_QUAD = Pipeline([('poly', PolynomialFeatures(degree = 2)),
                     ('linear', LinearRegression(fit_intercept = False))])
model_QUAD.fit(train[['horsepower']], train[['mpg']])
((test[['mpg']] - model_QUAD.predict(test[['horsepower']])) ** 2).mean()

model_CUBIC = Pipeline([('poly', PolynomialFeatures(degree = 3)),
                     ('linear', LinearRegression(fit_intercept = False))])
model_CUBIC.fit(train[['horsepower']], train[['mpg']])
((test[['mpg']] - model_CUBIC.predict(test[['horsepower']])) ** 2).mean()

# LOOCV
scores = cross_val_score(model_LR, autoData[['horsepower']],
                         autoData[['mpg']],
                         cv = 392, 
                         scoring = 'neg_mean_squared_error')
scores = (scores ** 2).mean() ** 0.5

predict = cross_val_predict(model_LR, autoData[['horsepower']],
                         autoData[['mpg']],
                         cv = 392)
mean_squared_error(autoData[['mpg']], predict)

model_scores = [];
for i in range(1, 6):
    model_GLR = Pipeline([('poly', PolynomialFeatures(degree = i)),
                          ('linear', LinearRegression(fit_intercept = False))])
    model_GLR.fit(autoData[['horsepower']], autoData[['mpg']])
    
    pred = cross_val_predict(model_GLR, autoData[['horsepower']],
                      autoData[['mpg']],
                      cv = 392)
    model_scores.append(mean_squared_error(autoData[['mpg']], pred))
    
# BOOTSTRAP
def bootstrap_reg(data, n, sample_size, deg):
    error = []
    for i in range(0, n):
        model = Pipeline([('poly', PolynomialFeatures(degree = deg)),
                          ('linear', LinearRegression(fit_intercept = False))])
        dataSample = data.sample(sample_size, replace = True)
        model.fit(dataSample[[0]], dataSample[[1]])
        error.append(((dataSample[[1]] - model.predict(dataSample[[0]])) ** 2).mean()[0])
    return(error)
    
def alpha_boot(data, n, sample_size):
    alpha = [] # empty array to hold statistics from each iteration
    for i in range(0, n):
        dataSample = data.sample(sample_size, replace = True) #sample data with replacement
        x = dataSample[[0]]
        y = dataSample[[1]]
        alpha.append((np.var(y) - np.cov(x, y)) / (np.var(x)+np.var(y)- 2 * np.cov(x, y)))
    return(alpha)

dataPort = pd.read_csv('../data/Portfolio.csv', index_col = 0)
testAlpha = alpha_boot(dataPort, 1000, 100)

dataSample = dataPort.sample(100, replace = True) #sample data with replacement
x = dataSample["X"]
y = dataSample["Y"]
testcov = np.cov(x, y)[0, 1]
    
model_BS = bootstrap_reg(autoData, 1000, 100, 1)
np.mean(model_BS)
np.var(mode_BS)

model_BS = bootstrap_reg(autoData, 1000, 100, 2)
np.mean(model_BS)
np.var(model_BS)

# DEFAULT
defaultData = pd.read_csv('../data/Default.csv', index_col = 0)
enc = LabelEncoder()
defaultData.default = enc.fit_transform(defaultData.default)
defaultData.student = enc.fit_transform(defaultData.student)

train, test = train_test_split(defaultData, test_size = 0.2, random_state = 1)
dLG = linear_model.LogisticRegression()
dLG.fit(train[['balance', 'student']], train['default'])
predLG = dLG.predict(test[['balance', 'student']])
conLG = confusion_matrix(test['default'], predLG)
scorLG = dLG.score(test[['balance', 'student']], test['default'])
