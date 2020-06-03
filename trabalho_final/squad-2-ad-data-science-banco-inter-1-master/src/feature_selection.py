## feature_selection

import numpy as np
import pandas as pd
from config import Configure
from preprocessing import Preprocessing

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class FeatureSelection:
    def __init__(self, mkt,  df1, df2, df3, pp_params, fs_params):
        self.mkt = mkt
        self.df1 = df1
        self.df2 = pd.merge(df2, mkt, on='id', how='inner')
        self.df3 = pd.merge(df3, mkt, on='id', how='inner')
        self.fs_params = fs_params
        self.pp_params = pp_params
        self.X = []
        self.y = []


    def eda_nan_columns_drop(self):
        # null reg per column
        nan_num = (self.mkt.isnull().sum(axis=0)) / self.mkt.shape[0]
        # get columns whose have more nulls than defined threshold
        column_nans = self.mkt.columns[nan_num >= self.fs_params['threshold']]
        #self.columns_chosen = list(params['EDA']) + list(column_nans)
        return list(self.fs_params['EDA']) + list(column_nans)

    def drop_columns(self):
        self.df1['target'] = np.ones(self.df1.shape[0]).astype(np.int)
        self.df2['target'] = 2*np.ones(self.df2.shape[0]).astype(np.int)
        self.df3['target'] = 3*np.ones(self.df3.shape[0]).astype(np.int)
        concat_df = [self.df1, self.df2, self.df3]
        concat_df = pd.concat(concat_df, ignore_index=True)
        concat_df.drop(self.eda_nan_columns_drop()+['Unnamed: 0_x', 'Unnamed: 0_y', 'id'],
                       axis=1,
                       inplace=True)
        self.X = concat_df.drop(['target'], axis=1)
        self.y = concat_df['target']

    def feature_selection_algorithm(self, m):
        self.eda_nan_columns_drop()
        self.drop_columns()
        manual_features = self.pp_params['manual_encoding'].keys()
        columns_type = self.X.dtypes
        cat_features = columns_type[(columns_type == 'object')].keys()
        cat_features = cat_features.drop(manual_features)
        bool_features = columns_type[(columns_type == 'bool')].keys()
        num_features = columns_type[(columns_type != 'object') & (columns_type != 'bool')].keys()
        self.X[bool_features] = self.X[bool_features].astype('category')
        pre_process = Preprocessing(cat_vars=cat_features,
                                    num_vars=num_features,
                                    bool_vars=bool_features,
                                    manual_vars=manual_features)

        return pre_process.feature_selection_apply(self.X, self.y, method=m)

    # def feature_selection_method():
    #     #data = FeatureSelection(df1=self.df1 df2=self.df2, df3=self.df3, mkt=self.mkt)
    #     #data.eda_nan_columns(settings.feature_selection_params)
    #     #data = data.drop_columns()
    #     #y = data['target'].values
    #     #X = data.drop(['target', 'Unnamed: 0'], axis=1)
    #     columns_type = self.X.dtypes
    #     manual_features = ['de_saude_tributaria', 'de_nivel_atividade']
    #     cat_features = columns_type[(columns_type == 'object')].keys()
    #     cat_features = cat_features.drop(manual_features)
    #     bool_features = columns_type[(columns_type == 'bool')].keys()
    #     num_features = columns_type[(columns_type != 'object') & (columns_type != 'bool')].keys()
    #     X[bool_features] = X[bool_features].astype('category')
    #     pre_process = Preprocessing(cat_vars=cat_features,
    #                                 num_vars=num_features,
    #                                 bool_vars=bool_features,
    #                                 manual_vars=manual_features)
    #     return mkt[pre_process.feature_selection_apply(X, y).values]

