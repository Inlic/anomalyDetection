
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