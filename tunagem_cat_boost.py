import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from tratamento import Tratamento
import pickle

# Carregando os dados

diretorio = 'dados'

with open(diretorio + '/dados_tratados.pkl', 'rb') as f:
    dados_tratados = pickle.load(f)

x_train = dados_tratados['x_train']
y_train = dados_tratados['y_train']

x_train = x_train.values
y_train = y_train.values

print('Dowload dos dados com sucesso!')

## Parametros random forest
parametros_cat = {"iterations": [2, 10, 20, 30, 50, 100, 200, 300, 400, 500],
                    "learning_rate": [0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1],
                    "depth": [1, 3, 5, 7, 9,10],
                    'l2_leaf_reg': [1, 3, 5, 7, 9],
                }
grid_random = RandomizedSearchCV(
            CatBoostRegressor(verbose=0), 
            parametros_cat,
            verbose=0,
            scoring="neg_root_mean_squared_error"
        )
grid_random.fit(x_train, y_train)

diretorio = 'modelos'

with open(diretorio + '/cat_boost_tunado.pkl', 'wb') as f:
    pickle.dump(grid_random, f)

print('Modelo Tunado com sucesso!')