import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from tratamento import Tratamento
import pickle
import tensorflow as tf
import time
import shap
from shap import Cohorts, Explanation
from shap.utils._exceptions import DimensionError
from sklearn.linear_model import LinearRegression,SGDRegressor,QuantileRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import keras
from keras import layers

##Avaliacao
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error, explained_variance_score,root_mean_squared_error,mean_absolute_percentage_error,root_mean_squared_log_error

from tratamento import Tratamento

import warnings

warnings.filterwarnings('ignore')

# Carregando os dados

diretorio = 'dados'

with open(diretorio + '/dados_tratados.pkl', 'rb') as f:
    dados_tratados = pickle.load(f)

x_train = dados_tratados['x_train']
y_train = dados_tratados['y_train']
y_test = dados_tratados['y_test']
x_test = dados_tratados['x_test']

x_train_colunms = x_train.copy()
x_test_columns = x_test.copy()

print('Dowload dos dados com sucesso!')

dados_importancia = pickle.load(open(diretorio + '/dados_importancia.pkl', 'rb'))
explainer = dados_importancia['explainer']
shap_values = dados_importancia['shape_values']
subset_x_treino = dados_importancia['subset_x_treino']
subset_y_treino = dados_importancia['subset_y_treino'] 

print('Shape Values carregados com sucesso!')


# Supondo que shap_values já esteja definido como no código original
def df_shap_importancia(shap_values):
    # Convert Explanation objects to dictionaries
    if isinstance(shap_values, Explanation):
        cohorts = {"": shap_values}
    elif isinstance(shap_values, Cohorts):
        cohorts = shap_values.cohorts
    elif isinstance(shap_values, dict):
        cohorts = shap_values
    else:
        raise TypeError("The shap_values argument must be an Explanation object, Cohorts object, or dictionary of Explanation objects!")

    # Desempacotando os Explanation objects
    cohort_labels = list(cohorts.keys())
    cohort_exps = list(cohorts.values())

    for i, exp in enumerate(cohort_exps):
        if not isinstance(exp, Explanation):
            raise TypeError("The shap_values argument must be an Explanation object, Cohorts object, or dictionary of Explanation objects!")
        
        if len(exp.shape) == 2:
            # Colapsa as arrays de Explanation para o formato (#features,)
            cohort_exps[i] = exp.abs.mean(0)

        if cohort_exps[i].shape != cohort_exps[0].shape:
            raise DimensionError("Quando passar vários Explanation objects, eles devem ter o mesmo número de colunas de features!")

    # Pega os nomes dos recursos e valores de SHAP
    feature_names = cohort_exps[0].feature_names
    values = np.array([cohort_exps[i].values for i in range(len(cohort_exps))])

    # Verifica se não está vazio
    if len(values[0]) == 0:
        raise ValueError("O Explanation fornecido está vazio, então não há nada para extrair.")

    # Cria um DataFrame com as importâncias médias dos SHAP values
    df_importancias = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Value': values.mean(axis=0)  # Média dos valores de SHAP para cada feature
    })

    # Ordena por ordem decrescente de importância
    df_importancias = df_importancias.sort_values(by='SHAP Value', ascending=False)

    # Mostra o DataFrame
    return df_importancias

importancia_shap = df_shap_importancia(shap_values)

print(importancia_shap.head(10))

diretorio = 'dados'

with open(diretorio + '/importancia_carcteristicas.pkl','wb') as f:
    pickle.dump(importancia_shap, f)


modelos = {
        'Linear': LinearRegression(),
        'Decision_tree':DecisionTreeRegressor(),
        'RandomForest':RandomForestRegressor(),
        'GradientBoosting':GradientBoostingRegressor(),
        'KNeighbors':KNeighborsRegressor(),
        'SVR':SVR(),
        'XGBR':XGBRegressor(),
        'Quantile':QuantileRegressor(),
        'CatBoost':CatBoostRegressor(),
        'SGDR':SGDRegressor()
    }
colunas_nivel = {}
modelos_base_treinados = {}
predict_modelos = {}
for num_colunms in range(10,len(importancia_shap)):
    ## inicio do tempo
    inicio = time.time()
    x_train_selection = x_train_colunms[importancia_shap.head(num_colunms)['Feature'].to_list()]
    x_test_selection = x_test_columns[importancia_shap.head(num_colunms)['Feature'].to_list()]
    print(x_train_selection.columns.to_list())
    colunas_nivel[num_colunms] = x_train_selection.columns.to_list()
    x_train_selection = x_train_selection.values
    x_test_selection = x_test_selection.values
    
    input_shape = x_train_selection.shape[1]
    
    model_rede = keras.Sequential()
    model_rede.add(layers.Dense(units=40, activation='relu', input_shape=(input_shape,)))
    model_rede.add(layers.Dense(units=60, activation='relu'))
    model_rede.add(layers.Dense(units=40, activation='relu'))
    model_rede.add(layers.Dense(units=1, activation='linear'))  # saída para regressão

    model_rede.compile(optimizer="adam", 
                                loss='mse',
                                metrics=[keras.metrics.RootMeanSquaredError()])
    
    modelos['Redes_neurais'] = model_rede
    
    for name, model in modelos.items():
        
        if name == 'Redes_neurais':   
            epochs_hist = model.fit(x_train_selection, 
                                    y_train, 
                                    epochs=10, 
                                    batch_size=200, 
                                    verbose=0)
                
            modelos_base_treinados[name+'_'+str(num_colunms)] = model
            y_pred = model.predict(x_test_selection).squeeze()
            predict_modelos[name+'_'+str(num_colunms)] = y_pred
            
            fim = time.time()
            tempo_execucao = round((fim - inicio) / 60,2)
            print(f"Tempo de execução {tempo_execucao} do modelo: {name+'_'+str(num_colunms)}")
            
        else:    
            if name == 'CatBoost':
                model_treinado = model.fit(x_train_selection, y_train, verbose=0)
            else:
                model_treinado = model.fit(x_train_selection, y_train)
                
            modelos_base_treinados[name+'_'+str(num_colunms)] = model
            y_pred = model.predict(x_test_selection)
            predict_modelos[name+'_'+str(num_colunms)] = y_pred
            fim = time.time()
            tempo_execucao = round((fim - inicio) / 60,2)
            print(f"Tempo de execução {tempo_execucao} do modelo: {name+'_'+str(num_colunms)}")
            
diretorio = 'modelos'

with open(diretorio + '/modelos_teste_caracteristicas.pkl', 'wb') as f:
    pickle.dump(modelos_base_treinados, f)



model_metrics_padrao: dict[str, list] = {
    'Model' : [],
    "RMSE" : [],
    "MSE" :[],
    "MAE" : [],
    'MAPE': [],
    'R2' : [],
    'ExplainedVariance' : [],  
    }
for name, preds in predict_modelos.items():
    model_metrics_padrao["Model"].append(name)
    model_metrics_padrao["RMSE"].append(root_mean_squared_error(y_test, preds))
    model_metrics_padrao["MSE"].append(mean_squared_error(y_test, preds))
    model_metrics_padrao["MAE"].append(mean_absolute_error(y_test, preds))
    model_metrics_padrao["R2"].append(r2_score(y_test, preds))
    model_metrics_padrao["MAPE"].append(mean_absolute_percentage_error(y_test, preds))
    model_metrics_padrao["ExplainedVariance"].append(explained_variance_score(y_test, preds))
    #model_metrics_padrao['RMSLE'].append(root_mean_squared_log_error(y_test, preds))


results_padrao = pd.DataFrame(model_metrics_padrao).sort_values(by='RMSE')
print(results_padrao)

diretorio = 'dados'
with open(diretorio + '/resultados_teste_caracterisricas.pkl','wb') as f:
    pickle.dump(results_padrao, f)

with open(diretorio + '/predicoes_teste_caracterisricas.pkl','wb') as f:
    pickle.dump(predict_modelos, f)