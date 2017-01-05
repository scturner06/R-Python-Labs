# IMPORTS
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats

# LOAD DATA
from ggplot import mtcars as myData

# head is much better
myData.head()
myData.iloc[:9]

# IMPORT csv
myData = pd.DataFrame.from_csv('./mtcars.csv')
myData.head()
myData.iloc[:9]

# Not exactly a python equilvent of str() but describe() and dtypes are close.
# Similarly for names() but list() in Python does the same more or less.
myData.shape
myData.describe()
myData.dtypes
list(myData)

# Will do the factor split on gears
myData = myData.assign(gear=myData.gear.astype("category"))

# No favstats() equilivent I know of and I don't use it in R anyway.
# Will instead compute each statistic independently.
myData[['mpg']].mean()
myData[['hp']].std()
myData[['wt']].median()
myData[['disp']].quantile()

# PLOTS

# scatter
plt.plot(myData[['cyl']], myData[['mpg']], 'ro')
plt.axis([3.8, 8.2, 9.8, 35.2])
plt.xlabel('cyl')
plt.ylabel('mpg')
plt.show()

# boxplot
myData = myData[['cyl']].assign(cyl=myData.cyl.astype("category"))
plt.boxplot(myData)
plt.show()