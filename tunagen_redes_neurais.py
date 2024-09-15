import pandas as pd
import numpy as np
import keras
import keras_tuner
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

def build_model(hp,camadas_ocultas=4):
    model = keras.Sequential()
    # Primeira camada com input_shape especificado corretamente
    model.add(keras.Input(shape=(x_train.shape[1],))) 
    
    model.add(keras.layers.Dense(
          hp.Choice('units', [8, 16, 32,50,64,100]),
          activation='relu'))
    # Camadas ocultas adicionais
    for _ in range(camadas_ocultas):
        model.add(keras.layers.Dense(
            hp.Choice('units', [8, 16, 32,50, 64,100]),
            activation='relu'))
    # Camada de saída
    model.add(keras.layers.Dense(1, activation='linear'))
    # Definição do otimizador
    opt = keras.optimizers.Adam(learning_rate= hp.Choice("learning_rate", [0.001, 0.01, 0.02 ,0.05,0.1, 0.2, 0.3]))
    # Compilação do modelo
    model.compile(loss=keras.losses.MeanSquaredError(), 
                  optimizer=opt, 
                  metrics=[keras.metrics.RootMeanSquaredError()])
    return model

tuner = keras_tuner.RandomSearch(
    build_model,
    objective=keras_tuner.Objective('root_mean_squared_error', direction='min'),
    max_trials=5,)

tuner.search(x_train, y_train, epochs=10,verbose=0)
best_model = tuner.get_best_models()[0]

diretorio = 'modelos'

best_model.save(diretorio+"/redes_neurais_tunado.h5")

print('Modelo Tunado com sucesso!')
