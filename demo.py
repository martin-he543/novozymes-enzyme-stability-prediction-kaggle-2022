# lib
import sys

# lib
import numpy as np
import pandas as pd 
import os
import plotly.express as px
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from tqdm import tqdm
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
from matplotlib_venn import venn3, venn3_circles
import torch
from biopandas.pdb import PandasPdb
from transformers import BertModel, BertTokenizer
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from sklearn import model_selection as sk_model_selection
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error,roc_auc_score,precision_score
from sklearn import metrics
import optuna
from boostaroota import BoostARoota
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_log_error
from optuna.samplers import TPESampler
import functools
from functools import partial
import xgboost as xgb
import joblib
from xgboost import plot_tree
import shap
from sklearn.metrics import mean_squared_error,roc_auc_score,precision_score
from sklearn import metrics
from Levenshtein import distance as levenshtein_distance

#import data
df_train = pd.read_csv('novozymes-enzyme-stability-prediction/train.csv')
df_update =  pd.read_csv('novozymes-enzyme-stability-prediction/train_updates_20220929.csv')
df_test = pd.read_csv('novozymes-enzyme-stability-prediction/test.csv')
df_sub = pd.read_csv('novozymes-enzyme-stability-prediction/sample_submission.csv')
wild = PandasPdb().read_pdb('novozymes-enzyme-stability-prediction/wildtype_structure_prediction_af2.pdb')

# df_train prep
df_train.head()
print(df_train.shape, df_train.protein_sequence.nunique())
(df_train.isnull().sum()/df_train.shape[0]*100).reset_index().rename(columns = {'index':'Columns in train data', 0:'% Nulls'})
df_train['pH'].value_counts()
df_train['data_source'].value_counts()
#train_update 
df_update.head()
print(df_update[df_update['protein_sequence'].notna()].shape)
df_update[df_update['protein_sequence'].notna()]
df_train = df_train[~df_train['seq_id'].isin(df_update[df_update['protein_sequence'].isnull()]['seq_id'].tolist())].reset_index(drop = True)
print(df_train.shape)

temp = df_update[df_update['protein_sequence'].notna()].reset_index(drop = True)[['seq_id','pH','tm']]
temp.columns = ['seq_id','updated_pH', 'updated_tm']
df_train = df_train.merge(temp, how = 'left', on = 'seq_id')
df_train['pH'] = np.where(df_train['updated_pH'].isnull(), df_train['pH'], df_train['updated_pH']) 
df_train['tm'] = np.where(df_train['updated_tm'].isnull(), df_train['tm'], df_train['updated_tm']) 
print(df_train[df_train['updated_pH'].notna()])
df_train.drop(columns = ['updated_pH', 'updated_tm'], inplace = True)
print(df_train.shape)

#test data
print(df_test.shape)
df_test.head()

#pdb
type(wild)
wild.df['ATOM'].info()
wild.df['ATOM'].groupby('residue_number').first()
wild.df['HETATM']
wild.df['ANISOU']
wild.df['OTHERS']

# protein seq wtih multiple pH
temp = df_train.groupby(['protein_sequence']).agg({'seq_id':'nunique'}).reset_index().rename(columns = {'seq_id':'# Rows'}).sort_values(by = '# Rows', ascending = False).reset_index(drop = True)
temp[temp['# Rows']>1]
df_train[df_train['protein_sequence']==temp[temp['# Rows']>1]['protein_sequence'].iloc[0]]
df_train.shape, df_train[['protein_sequence','pH']].drop_duplicates().shape
df_train[df_train['# Unique tm values'].notna()].shape[0] - df_train[df_train['# Unique tm values'].notna()][['protein_sequence','pH','tm']].drop_duplicates().shape[0]
