import numpy as np
import pandas as pd
from config import Configure
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt


# Custom Transformer that extracts columns passed as argument to its constructor
class FeatureSelector(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, feature_names):
        self._feature_names = feature_names

        # Return self nothing else to do here

    def fit(self, X, y=None):
        return self

        # Method that describes what we need this transformer to do

    def transform(self, X, y=None):
        return X[self._feature_names]


class ManualEncoding(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, feature_names=[]):
        self._feature_names = feature_names

        # Return self nothing else to do here

    def fit(self, X, y=None):
        return self

    def set_manual_transformation(self):
        settings = Configure()
        settings.set_pre_processing_params()
        rules = settings.pre_processing_params['manual_encoding']
        return rules

        # Method that describes what we need this transformer to do

    def transform(self, X, y=None):
        x = X.copy()
        dictionary = self.set_manual_transformation()
        for key in dictionary.keys():
            x.loc[:, key] = x[key].map(dictionary[key])

        return x.values





from config import Configure


class Preprocessing:
    def __init__(self, cat_vars, num_vars, bool_vars, manual_vars):
        self.cat_vars = cat_vars
        self.num_vars = num_vars
        self.bool_vars = bool_vars
        self.manual_vars = manual_vars


    def pipe_lines_creating(self):
        categorical_pipeline = Pipeline(steps=[('cat_selector', FeatureSelector(self.cat_vars)),
                                               ('cat_imputer', SimpleImputer(strategy='most_frequent')),
                                               ('cat_encoder', OrdinalEncoder())])
        manual_pipeline = Pipeline(steps=[('manual_selector', FeatureSelector(self.manual_vars)),
                                          ('manual_encoder', ManualEncoding()),
                                          ('manual_imputer', SimpleImputer(strategy='most_frequent'))])
        bool_pipeline = Pipeline(steps=[('bool_selector', FeatureSelector(self.bool_vars)),
                                        ('bool_imputer', SimpleImputer(strategy='most_frequent'))])
        num_pipeline = Pipeline(steps=[('num_selector', FeatureSelector(self.num_vars)),
                                       ('num_imputer', SimpleImputer(strategy='median')),
                                       ('scaler', StandardScaler())])
        return FeatureUnion(transformer_list=[('manual_pipeline', manual_pipeline),
                                                   ('categorical_pipeline', categorical_pipeline),
                                                   ('bool_pipeline', bool_pipeline),
                                                   ('numerical_pipeline', num_pipeline)])

    def feature_selection_apply(self, X, y, method):
        columns = X.columns
        feature_pipeline = self.pipe_lines_creating()
        x = feature_pipeline.fit_transform(X)
        if method == 'RFECV':
            clf = RandomForestClassifier(random_state=101)
            rfe_cv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(10), scoring='accuracy')
            rfe_cv.fit(x, y)
            important_columns = columns[rfe_cv.support_]
            dset = pd.DataFrame()
            dset['attr'] = important_columns
            dset['importance'] = rfe_cv.estimator_.feature_importances_
            dset = dset.sort_values(by='importance', ascending=False)
            dset = dset[dset['importance'] > 0.01]
            scores = rfe_cv.grid_scores_
            attr = dset['attr']
        elif method == 'LASSO':
            from metodo_eilert_V3 import metodo_eilert
            from sklearn.linear_model import LogisticRegression
            x = pd.DataFrame(x, columns=columns)
            x.drop('Unnamed: 0', axis =1, inplace=True)
            clf = LogisticRegression(solver='lbfgs')
            df = metodo_eilert(x,
                               y.values,
                               clf,
                               x.shape[1],
                               "AUROC")
            scores = df['AUROC'].values
            scores = scores[0:scores.argmax()]
            attr = df['Variavel'].values[0:scores.argmax()]
        else:
            print('Invalid method')
            scores = []
            attr = []
        return scores, attr



