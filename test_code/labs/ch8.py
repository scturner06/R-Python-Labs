# NOTES
# Q7) Sqrt behaves differently.  Will return to this later.
# Q8) Carseats data missing.  Added.  Parameters adjusted to better approximate
#     R results, but note that R and Python use different criteria by default
#     and when performing tree regression.
#     Pruning is not available in Python.  From their user guide
#     "Decision-tree learners can create over-complex trees that do not 
#     generalise the data well. This is called overfitting. Mechanisms such as 
#     pruning (not currently supported), setting the minimum number of samples 
#     required at a leaf node or setting the maximum depth of the tree are 
#     necessary to avoid this problem. 
#     Part c omited due to no pruning.
# Q9) b) It is quite difficult to get that level of detail in Python.  It involes
#     writing algorithms to dig through the udnerlying "tree" object which I
#     don't think would be very instructive.  Link for how to retrieve detailed
#     node info (http://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html)
#     Parts f-k omited due to no pruning.
# Q10) Hitters datat missing.  Added.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error

dataBoston = pd.read_csv('../../data/Boston.csv', index_col = 0)

train, test = train_test_split(dataBoston, test_size = 0.5, random_state = 1)

# Note that we must make Y a 1-D Array in order to work with sklearn RandomForestClassifer()
train_x = train.ix[:, 0:13]
test_x = test.ix[:, 0:13]
train_y = np.asarray(train['medv'], dtype = "float")
test_y = np.asarray(test['medv'], dtype = "float")

# We want to plot MSE versus the number of trees.  To do this in Python, we must construct a for loop
n_trees = 500

rf1 = RandomForestRegressor(warm_start = True)
boston1_mse = []
for i in range(1, n_trees + 1):
    rf1.set_params(n_estimators = i, max_features = "auto", random_state = 1)
    rf1.fit(train_x, train_y)
    boston1_mse.append(mean_squared_error(test_y, rf1.predict(test_x)))
    
rf2 = RandomForestRegressor(warm_start = True)
boston2_mse = []
for i in range(1, n_trees + 1):
    rf2.set_params(n_estimators = i, max_features = 0.5, random_state = 1)
    rf2.fit(train_x, train_y)
    boston2_mse.append(mean_squared_error(test_y, rf2.predict(test_x)))

rf3 = RandomForestRegressor(warm_start = True)
boston3_mse = []
for i in range(1, n_trees + 1):
    rf3.set_params(n_estimators = i, max_features = "sqrt", random_state = 1)
    rf3.fit(train_x, train_y)
    boston3_mse.append(mean_squared_error(test_y, rf3.predict(test_x)))
    
plt.plot(boston1_mse, 'g', label = 'm = p')
plt.plot(boston2_mse, 'r', label = 'm = p/2')
plt.plot(boston3_mse, 'b', label = 'm = sqrt(p)')
plt.ylim(10, 19)
plt.xlabel('Number of Trees')
plt.ylabel('MSE')
plt.legend()
plt.show()

# OMITED
from sklearn.model_selection import cross_val_predict

n = 20

cvScore = []
for i in range(2, n + 1):
    tree = DecisionTreeRegressor(max_leaf_nodes = i, min_samples_split = 10, min_samples_leaf = 5)
    tree.fit(train.ix[:, 1:13], train['Sales'])
    pred = cross_val_predict(tree, dataCarseats.ix[:, 1:13], dataCarseats['Sales'], cv = 10)
    cvScore.append(mean_squared_error(dataCarseats['Sales'], pred))
    #cvScore.append(np.mean(cross_val_pred(tree, dataCarseats.ix[:, 1:13], dataCarseats['Sales'], cv = 10)))
    

# Q9
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

dataOJ = pd.read_csv('../../data/OJ.csv', index_col = 0)
dataOJ.dtypes

from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
dataOJ.Store7 = enc.fit_transform(dataOJ.Store7)
train, test = train_test_split(dataOJ, test_size = 0.25, random_state = 1)

from sklearn.tree import DecisionTreeClassifier
OJ_tree = DecisionTreeClassifier(min_samples_split = 10, min_samples_leaf = 5)
OJ_tree.fit(train.ix[:, 1:18], train['Purchase'])

OJ_tree.score(test.ix[:, 1:18], test['Purchase'])

import pydotplus
from sklearn import tree
from IPython.display import Image

dot_data = tree.export_graphviz(OJ_tree, out_file=None, 
                         feature_names=train.columns[1:18],  
                         class_names=train['Purchase'],  
                         filled=True, rounded=True,  
                         special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

from sklearn.metrics import confusion_matrix
confusion_matrix(test['Purchase'], OJ_tree.predict(test.ix[:, 1:18]))
1 - (135 + 78)/(135 + 21 + 34 + 78)

# Q10
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

dataHitters = pd.read_csv('../../data/Hitters.csv', index_col = 0)
dataHitters = dataHitters.dropna()
dataHitters['Salary'] = np.log(dataHitters['Salary'])

enc = LabelEncoder()
dataHitters.League = enc.fit_transform(dataHitters.League)
dataHitters.Division = enc.fit_transform(dataHitters.Division)
dataHitters.NewLeague = enc.fit_transform(dataHitters.NewLeague)

train, test = train_test_split(dataHitters, test_size = 63, random_state = 1)

train_x = train.drop('Salary', axis = 1)
test_x = test.drop('Salary', axis = 1)
train_y = train['Salary']
test_y = test['Salary']

#c
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

lamdas = 10 ** np.arange(-10, 0, 0.1)

train_err = []
for i in lamdas:
    gbr_hitters = GradientBoostingRegressor(n_estimators = 1000,
                                            learning_rate = i,
                                            random_state = 0)
    gbr_hitters.fit(train_x, train_y)
    train_err.append(np.mean(gbr_hitters.train_score_))
    
plt.plot(lamdas, train_err, '.-b')
plt.ylabel("Train Loss")
plt.xlabel("Learning Rate")
plt.show()

test_err = []
for i in lamdas:
    gbr_hitters = GradientBoostingRegressor(n_estimators = 1000,
                                            learning_rate = i,
                                            random_state = 0)
    gbr_hitters.fit(train_x, train_y)
    test_err.append(mean_squared_error(test_y, gbr_hitters.predict(test_x)))
    
plt.plot(lamdas, test_err, '.-b')
plt.ylabel("Test MSE")
plt.xlabel("Learning Rate")
plt.show()

min(test_err)
lamdas[test_err.index(min(test_err))]

# e
from sklearn.linear_model import LinearRegression
lm_hitters = LinearRegression()
lm_hitters.fit(train_x, train_y)
mean_squared_error(test_y, lm_hitters.predict(test_x))

from sklearn.linear_model import Ridge
rg_hitters = Ridge(alpha = 0)
rg_hitters.fit(train_x, train_y)
mean_squared_error(test_y, rg_hitters.predict(test_x))

# f
gbr_hitters = GradientBoostingRegressor(n_estimators = 1000,
                                        learning_rate = 0.32,
                                        random_state = 1)
gbr_hitters.fit(train_x, train_y)
hitters_imp = pd.DataFrame({'feature':  train_x.columns
                            , 'score': gbr_hitters.feature_importances_
                            }).sort_values('score')

plt.barh(range(0, len(hitters_imp)), hitters_imp['score'],
        tick_label = hitters_imp['feature'])
plt.show()

# g
from sklearn.ensemble import RandomForestRegressor

rf_hitters = RandomForestRegressor(n_estimators = 500, random_state = 1)
rf_hitters.fit(train_x, train_y)
mean_squared_error(test_y, rf_hitters.predict(test_x))

# Q11
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

dataCaravan = pd.read_csv('../../data/Caravan.csv', index_col = 0)

enc = LabelEncoder()
dataCaravan.Purchase = enc.fit_transform(dataCaravan.Purchase)

train, test = train_test_split(dataCaravan, test_size = 4822, random_state = 0)

train_x = train.drop('Purchase', axis = 1)
test_x = test.drop('Purchase', axis = 1)
train_y = train['Purchase']
test_y = test['Purchase']

# b
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error

gbc_caravan = GradientBoostingClassifier(n_estimators = 1000, learning_rate = 0.01,
                                  random_state = 0)
gbc_caravan.fit(train_x, train_y)

caravan_imp = pd.DataFrame({'feature':  train_x.columns
                            , 'score': gbc_caravan.feature_importances_
                            }).sort_values('score')

plt.barh(range(0, len(caravan_imp)), caravan_imp['score'],
        tick_label = caravan_imp['feature'])
plt.show()

caravan_imp.sort_values('score', ascending = False)
# c
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

pred = []
for prob in gbc_caravan.predict_proba(test_x)[:, 1]:
    if(prob >= 0.2):
        pred.append(1)
    else:
        pred.append(0)

confusion_matrix(test_y, pred)
44/(205+44)

lgr_caravan = LogisticRegression()
lgr_caravan.fit(train_x, train_y)

pred = []
for prob in lgr_caravan.predict_proba(test_x)[:, 1]:
    if(prob >= 0.2):
        pred.append(1)
    else:
        pred.append(0)
        
confusion_matrix(test_y, pred)
55/(302+55)

# Q12
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

dataWeekly = pd.read_csv('../../data/Weekly.csv', index_col = 0)
dataWeekly = dataWeekly.drop(['Year', 'Today'], axis = 1)

enc = LabelEncoder()
dataWeekly.Direction = enc.fit_transform(dataWeekly.Direction)

train, test = train_test_split(dataWeekly, test_size = 0.5, random_state = 1)

train_x = train.drop('Direction', axis = 1)
test_x = test.drop('Direction', axis = 1)
train_y = train['Direction']
test_y = test['Direction']

# logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

lgr_weekly = LogisticRegression()
lgr_weekly.fit(train_x, train_y)

pred = lgr_weekly.predict(test_x)

confusion_matrix(test_y, pred)
lgr_weekly.score(test_x, test_y)

# boosting
from sklearn.ensemble import GradientBoostingClassifier

gbc_weekly = GradientBoostingClassifier(n_estimators = 5000, random_state = 0)
gbc_weekly.fit(train_x, train_y)

pred = gbc_weekly.predict(test_x)

confusion_matrix(test_y, pred)
gbc_weekly.score(test_x, test_y)

# bagging
from sklearn.ensemble import RandomForestClassifier

bag_weekly = RandomForestClassifier(n_estimators = 500, max_features = 6,
                                    random_state = 0)
bag_weekly.fit(train_x, train_y)

pred = bag_weekly.predict(test_x)

confusion_matrix(test_y, pred)
bag_weekly.score(test_x, test_y)

# random forest
bag_weekly = RandomForestClassifier(n_estimators = 500, max_features = 2,
                                    random_state = 0)
bag_weekly.fit(train_x, train_y)

pred = bag_weekly.predict(test_x)

confusion_matrix(test_y, pred)
bag_weekly.score(test_x, test_y)