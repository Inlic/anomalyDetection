
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from sklearn.ensemble import IsolationForest

data = pd.read_csv('example_wp_log_peyton_manning.csv', parse_dates=['ds'])

data.head()

data1 = data.copy()
data1['day'] = [i.weekday() for i in data.ds]
data1['month'] = data1.ds.dt.month
data1.head()

clf = IsolationForest(max_samples=100, contamination='auto')
clf.fit(data1[['y', 'month']])

y_pred_train = clf.predict(data1[['y','month']])

ax = data.iloc[y_pred_train == 1].plot(x='ds', y='y',kind='scatter',figsize=(15,8))
data.iloc[y_pred_train == -1].plot(x='ds', y='y', kind='scatter', color='red', ax=ax)
plot.title("Anomalies w/Isolation Forests")
plot.show()