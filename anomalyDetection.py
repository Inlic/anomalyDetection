
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from scipy.stats import norm
from sklearn.ensemble import IsolationForest

data = pd.read_csv('example_wp_log_peyton_manning.csv', parse_dates=['ds'])

bruce_data = pd.read_csv('learn_and_DS_play_data.csv',parse_dates=['DATE'])

data.head()
bruce_data.head()

bruce_data_copy = bruce_data.copy()
bruce_data_copy.head()

# #visualize the data
#
# bruce_data.describe()
#
# bruce_data_copy[["Asset1","Asset2"]].agg(['min','max','mean','std']).round(decimals=2)
#
# fig, ax = plot.subplots(1,1)
# # bruce_data_copy[["Asset1","Asset2"]].kde(ax=ax, legend=False,title='Histogram: Asset1 vs Asset2')
# # Histograms of the data
# bruce_data_copy[["Asset1","Asset2"]].plot.hist(grid=True, bins=100, ax=ax)
#
# # Creating a histogram of a normal distribution with mean 10 and std 1.
# dist = norm(loc=10, scale=1).rvs(size=2000)
# x = np.linspace(0,20,num=100)
#
# # Dropping all the plots on top of each other.
# ax.hist(dist, bins=100)
#
# ####

# data1 = data.copy()
bruce_data_copy['day'] = [i.weekday() for i in bruce_data.DATE]
bruce_data_copy['month'] = bruce_data_copy.DATE.dt.month
# data1.head()

# alt + shift + e for batch statements in python console

clf = IsolationForest(max_samples=100, contamination=0.05)
clf.fit(bruce_data_copy[['Asset1','month']])
scores1 = clf.decision_function(bruce_data_copy[['Asset1','month']])
asset1AnomalyMean = sum(scores1)/len(scores1)

y_pred_train = clf.predict(bruce_data_copy[['Asset1','month']])

ax = bruce_data.iloc[y_pred_train == 1].plot(x='DATE', y='Asset1',kind='scatter',figsize=(15,8))
bruce_data.iloc[y_pred_train == -1].plot(x='DATE', y='Asset1', kind='scatter', color='red', ax=ax)
plot.title("Anomalies w/Isolation Forests Asset 1")
plot.show()



#############################  Anomaly Scores Lower = more abnormal

clf = IsolationForest(max_samples=100, contamination=0.05)
clf.fit(bruce_data_copy[['Asset2','month']])
scores2 = clf.decision_function(bruce_data_copy[['Asset2','month']])
asset2AnomalyMean = sum(scores2)/len(scores2)


y_pred_train = clf.predict(bruce_data_copy[['Asset2','month']])

ax = bruce_data.iloc[y_pred_train == 1].plot(x='DATE', y='Asset2',kind='scatter',figsize=(15,8))
bruce_data.iloc[y_pred_train == -1].plot(x='DATE', y='Asset2', kind='scatter', color='red', ax=ax)
plot.title("Anomalies w/Isolation Forests Asset 2")
plot.show()

len(bruce_data.iloc[y_pred_train == 1])

# clf = IsolationForest(max_samples=100, contamination=.5)
# clf.fit(data1[['y', 'month']])
#
# y_pred_train = clf.predict(data1[['y','month']])

# ax = data.iloc[y_pred_train == 1].plot(x='ds', y='y',kind='scatter',figsize=(15,8))
# data.iloc[y_pred_train == -1].plot(x='ds', y='y', kind='scatter', color='red', ax=ax)
# plot.title("Anomalies w/Isolation Forests")
# plot.show()

