from feature_selection import FeatureSelection
from config import Configure
import pandas as pd

print('\n Aplicando algoritmo de seleção de parâmetros')
settings = Configure()
settings.set_fs_params()
settings.set_pre_processing_params()
pp_params = settings.pre_processing_params
fs_params = settings.feature_selection_params
df1 = pd.read_csv(settings.pf1_folder)
df2 = pd.read_csv(settings.pf2_folder)
df3 = pd.read_csv(settings.pf3_folder)
mkt = pd.read_csv(settings.mkt_folder)
fs = FeatureSelection(mkt,  df1, df2, df3, pp_params, fs_params)
values, features = fs.feature_selection_algorithm(m='RFECV')
columns = features.values[features.values != 'Unnamed: 0']
mkt = mkt[columns]

