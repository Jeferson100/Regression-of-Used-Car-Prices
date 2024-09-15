import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
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

diretorio_models = 'modelos'
## Parametros random forest
parametros_random = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'max_depth': [1,2,60,70,80,90,100],
            'max_features': [1, 3, 4,5,6,8,10],
            'criterion':['squared_error', 'friedman_mse',],
            "learning_rate": [0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1],
                 }
grid_random = RandomizedSearchCV(
            GradientBoostingRegressor(), 
            parametros_random,
            verbose=0, 
            scoring="neg_root_mean_squared_error"
        )
grid_random.fit(x_train, y_train)


with open(diretorio_models + '/gradiente_boost_tunado.pkl', 'wb') as f:
    pickle.dump(grid_random, f)

print('Modelo Tunado com sucesso!')