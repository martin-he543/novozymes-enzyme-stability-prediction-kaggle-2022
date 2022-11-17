import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import spearmanr

df_train = pd.read_csv('sample_data/train.csv', index_col = 'seq_id')
df_test = pd.read_csv('sample_data/test.csv', index_col = 'seq_id')
train_updates = pd.read_csv('sample_data/train_updates_20220929.csv')
submission = pd.read_csv('sample_data/sample_submission.csv', index_col = 'seq_id')

df_train.drop(columns = 'data_source', inplace = True)
df_test.drop(columns = 'data_source', inplace = True)