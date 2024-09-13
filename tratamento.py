
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
import sys
sys.path.append('/home/vscode/.local/lib/python3.10/site-packages')
from feature_engine.transformation import LogCpTransformer, YeoJohnsonTransformer, BoxCoxTransformer


class Tratamento:
    """
    Esta classe contém métodos para tratamento e transformação de dados em um DataFrame.
    Inclui funções para remoção de colunas, extração de características como cilindros, marchas, potência (HP), 
    e manipulação de dados categóricos e numéricos.
    """
    def retirando_ids(self, dados):
        """
        Remove a coluna `id` de um DataFrame.

        - **Parâmetros:**
        - `dados` (pd.DataFrame): DataFrame que contém a coluna `id`.

        - **Retorna:**
        - `dados` (pd.DataFrame): DataFrame sem a coluna `id`.
        - `id_dados` (pd.Series): Série contendo os IDs removidos.
        """
        dados_sem_id = dados.copy()
        id_dados = dados_sem_id["id"]
        dados_sem_id.drop("id", axis=1, inplace=True)
        return dados_sem_id, id_dados

    def extrair_cilindros(self, dados):
        """
        Extrai o número de cilindros de uma coluna de motores e limpa os dados.

        - **Parâmetros:**
          - `dados` (pd.DataFrame): DataFrame contendo a coluna 'engine' com descrições de motores.

        - **Retorna:**
          - `dados_cilindros` (pd.DataFrame): DataFrame com uma nova coluna 'cilindros', contendo o número de cilindros.
        """
        dados_cilindros = dados.copy()
        lista_engine = []
        for t in dados_cilindros['engine'].str.split(' '):
            if 'HP' in t[0]:
                if any(x in t[2] for x in ['Straight', 'Flat', 'Cylinder']):
                    lista_engine.append(t[3])
                elif 'Electric' in t[0] or 'Electric' in t[1]:
                    lista_engine.append('0')
                else:
                    lista_engine.append(t[2])
            elif 'L' in t[0]:
                try:
                    if any(x in t[1] for x in ['Straight', 'Flat', 'Cylinder']):
                        lista_engine.append(t[2])
                    else:
                        lista_engine.append(t[1])
                except IndexError:
                    lista_engine.append('None')
            elif len(t) > 1:
                if 'Liter' in t[1]:
                    lista_engine.append('None')
                else:
                    lista_engine.append(t)
            elif 'Electric' == t[0]:
                lista_engine.append('0')
            elif 'Intercooled' == t[0]:
                lista_engine.append(t[3])
            elif 'Dual' == t[0]:
                lista_engine.append('None')
            else:
                lista_engine.append('None')

        dados_cilindros['cilindros'] = lista_engine
        # Limpeza de valores na coluna de cilindros
        dados_cilindros['cilindros'] = pd.to_numeric(dados_cilindros['cilindros'].str.replace(r'[VIHW]', '', regex=True)
                                        .str.replace('Electric', '0')
                                        .str.replace('Rotary', 'None'), errors='coerce')
        # Preencher valores nulos com 0 para veículos elétricos
        dados_cilindros.loc[(dados_cilindros['cilindros'].isna()) & (dados_cilindros['fuel_type'] == 'electric'), 'cilindros'] = 0

        return dados_cilindros
    
    def extrair_hp(self, dados):
        """
        Extrai a potência do motor (HP) a partir da coluna 'engine'.

        - **Parâmetros:**
          - `dados` (pd.DataFrame): DataFrame contendo a coluna 'engine'.

        - **Retorna:**
          - `dados_hp` (pd.DataFrame): DataFrame com uma nova coluna 'hp', contendo a potência do motor.
        """
        dados_hp = dados.copy()
        # potencia do motor
        dados_hp['hp'] = pd.to_numeric([float(i[0].split('HP')[0]) if 'HP' in i[0] else None for i in dados_hp['engine'].
                                     str.split(' ')],errors='coerce')
        return dados_hp
    
    def extrair_marchas(self,dados):
        """
        Extrai o número de marchas a partir da coluna 'transmission'.

        - **Parâmetros:**
          - `dados` (pd.DataFrame): DataFrame contendo a coluna 'transmission'.

        - **Retorna:**
          - `dados_marchas` (pd.DataFrame): DataFrame com uma nova coluna 'marchas', contendo o número de marchas.
        """
        dados_marchas = dados.copy()
        dados_marchas['marchas'] = pd.to_numeric([i[0] if len(i) > 1 else None for i in dados_marchas['transmission'].str.split('-')],errors='coerce')
        return dados_marchas
    
    def extrair_idade_carros(self,dados):
        """
        Calcula a idade dos carros com base no ano do modelo.

        - **Parâmetros:**
          - `dados` (pd.DataFrame): DataFrame contendo a coluna 'model_year'.

        - **Retorna:**
          - `dados_idade` (pd.DataFrame): DataFrame com uma nova coluna 'idade_carro', contendo a idade dos carros.
        """
        dados_idade = dados.copy()
        # Idade do carro em anos
        dados_idade['idade_carro'] = datetime.datetime.now().year - dados_idade.model_year
        return dados_idade
    
   
    def extrair_cambio(self,dados):
    
        """
        Classifica o tipo de câmbio (automático, dual, manual) a partir da coluna 'transmission'.

        - **Parâmetros:**
          - `dados` (pd.DataFrame): DataFrame contendo a coluna 'transmission'.

        - **Retorna:**
          - `dados_cambio` (pd.DataFrame): DataFrame com uma nova coluna 'cambio', contendo a classificação do câmbio.
        """
        
        dados_cambio = dados.copy()
        
        automatic_patterns = [
            'Automatic', 'Auto-Shift', 'DCT Automatic', 'CVT', 'A/T'
        ]

        # Cria a nova coluna 'automatico' com base nos padrões
        dados_cambio['cambio'] = np.where(
            dados_cambio['transmission'].str.contains('|'.join(automatic_patterns), case=False, na=False),
            'automatico',
            np.where(dados_cambio['transmission'].str.contains('Dual'),'dual','manual') )
        return dados_cambio
    
    def drop_columns(self,dados,columns_drop):
        """
        Remove colunas específicas de um DataFrame.

        - **Parâmetros:**
        - `dados` (pd.DataFrame): DataFrame do qual as colunas serão removidas.
        - `columns_drop` (list): Lista de colunas a serem removidas.

        - **Retorna:**
        - `dados` (pd.DataFrame): DataFrame sem as colunas especificadas.
        """
        dados_drop = dados.copy()
        
        dados_drop.drop(columns_drop, axis=1, inplace=True)
        return dados_drop

    def imputar_dados_faltantes_modelo(
        self, dados, classe_predictor, categorical_features, numerical_features, modelo_imputer=None,
    ):
        """
        Imputa dados faltantes utilizando um modelo de aprendizado.

        - **Parâmetros:**
          - `dados` (pd.DataFrame): DataFrame com dados faltantes.
          - `classe_predictor` (str): Coluna alvo a ser imputada.
          - `categorical_features` (list): Lista de colunas categóricas.
          - `numerical_features` (list): Lista de colunas numéricas.
          - `modelo_imputer` (opcional): Modelo já treinado para imputação.

        - **Retorna:**
          - `dados_inputar` (pd.DataFrame): DataFrame com os valores imputados.
          - `pipe` (Pipeline): Modelo utilizado para a imputação, se criado.
        """
        if modelo_imputer is None:
            dados_inputar = dados.copy()
            ## substituindo os valores faltantes de marchas
            train_marchas = dados_inputar[dados_inputar[classe_predictor].notnull()]
            x_train_marchas = train_marchas[categorical_features + numerical_features]
            y_train_marchas = train_marchas[classe_predictor]

            pip_cat = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='infrequent_if_exist'))
            ])

            pip_num = Pipeline(steps=[
                ('inputar', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # Criação do ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', pip_num, numerical_features),
                    ('cat', pip_cat, categorical_features)
                ]
            )
            # Criação do pipeline final
            pipe = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor())
            ])
                        

            pipe.fit(x_train_marchas, y_train_marchas)

            x_predict_marchas= dados_inputar.loc[dados_inputar[classe_predictor].isnull(),categorical_features + numerical_features]

            dados_inputar.loc[x_predict_marchas.index,classe_predictor] = pipe.predict(x_predict_marchas).round()
            
            return dados_inputar, pipe
        else:
            dados_predict = dados.copy()
            
            x_predict_marchas= dados_predict.loc[dados_predict[classe_predictor].isnull(),categorical_features + numerical_features]
            
            dados_predict.loc[x_predict_marchas.index,classe_predictor] = modelo_imputer.predict(x_predict_marchas).round()
            
            return dados_predict
        
    def custom_combiner(self, feature, category):
        """
        Combina o nome da feature e da categoria em uma string.

        - **Parâmetros:**
        - `feature` (str): Nome da feature.
        - `category` (str): Nome da categoria.

        - **Retorna:**
        - `str`: String combinada da feature e categoria.
        """
        return str(feature) + "_"  + str(category)
    
    def criando_onehot(self,dados, col_categorical, model_onehot=None, list_frequenci = None):
            """
            Aplica OneHotEncoder às colunas categóricas especificadas.

            - **Parâmetros:**
            - `dados` (pd.DataFrame): DataFrame com as colunas categóricas.
            - `col_categorical` (list): Lista de colunas categóricas a serem transformadas.
            - `model_onehot` (dict, opcional): Dicionário de modelos OneHotEncoder a serem utilizados.
            - `list_frequenci` (list, opcional): Frequência mínima para OneHotEncoding.

            - **Retorna:**
            - `dados` (pd.DataFrame): DataFrame transformado com as colunas OneHot adicionadas.
            - `modelos_onehots` (dict, opcional): Dicionário de modelos OneHotEncoder utilizados.
            """

            modelos_onehots = {}
            
            if list_frequenci == None: 
                list_frequenci=len(col_categorical) * [0.05]

            if model_onehot is None:
                for frequenci, col in zip(list_frequenci,col_categorical):
                    model_onehot = OneHotEncoder(feature_name_combiner=self.custom_combiner,
                                                handle_unknown='infrequent_if_exist',
                                                min_frequency=frequenci)
                    model_onehot.fit(dados[[col]])
                    dados_onehot = model_onehot.transform(dados[[col]])
                    dados_onehot_df = pd.DataFrame(dados_onehot.toarray(),
                                                columns=model_onehot.get_feature_names_out(),
                                                index=dados.index)
                    dados = dados.join(dados_onehot_df)
                    modelos_onehots[col] = model_onehot
                    dados.drop(col, axis=1, inplace=True)

                return dados, modelos_onehots

            else:
                for col in col_categorical:
                    dados_onehot = model_onehot[col].transform(dados[[col]])
                    dados_onehot_df = pd.DataFrame(dados_onehot.toarray(),
                                                columns=model_onehot[col].get_feature_names_out(),
                                                index=dados.index)
                    dados = dados.join(dados_onehot_df)
                    dados.drop(col, axis=1, inplace=True)

                return dados
            
    def tipos_variaveis_numericas(self, dados, tipo=None):
        """
        Aplica transformação numérica aos dados.

        Parâmetros:
        - dados (array-like): Dados a serem transformados.
        - tipo (str, opicional): Tipo de transformação a ser aplicada. Pode ser 'StandardScaler', 'MinMaxScaler', 'BoxCoxTransformer', 'LogCpTransformer' ou 'YeoJohnsonTransformer'. Se não for fornecido, será utilizado 'StandardScaler' como padrão.

        Retorno:
        - dados_transformados (array-like): Dados transformados.
        - transformador (objeto): Instância do transformador utilizado.
        """

        # Define os transformadores disponíveis
        transformadores = {
            'StandardScaler': StandardScaler,
            'MinMaxScaler': MinMaxScaler,
            'BoxCoxTransformer': BoxCoxTransformer,
            'LogCpTransformer': LogCpTransformer,
            'YeoJohnsonTransformer': YeoJohnsonTransformer
        }

        # Verifica se o tipo de transformação é válido
        if tipo is None:
            tipo = 'StandardScaler'
        elif tipo not in transformadores:
            raise ValueError("Tipo de transformação inválido")

        # Instancia o transformador
        transformador = transformadores[tipo]()

        # Aplica a transformação
        dados_transformados = transformador.fit_transform(dados)

        return dados_transformados, transformador

    def transformacao_variaveis_numericas(self, dados, col_transformada, tipo=None, modelo_imputer=None):
        """
        Aplica transformação numérica a colunas específicas de DataFrames .

        - **Parâmetros:**
        - `dados` (pd.DataFrame): Conjunto de dados.
        - `col_numeric` (list): Lista de colunas numéricas a serem transformadas.
        - `tipo` (str, opcional): Tipo de transformação a ser aplicada (padrão: 'StandardScaler').
        - `modelo_imputer` (dict, opcional): Dicionário de modelos de imputação para o conjunto de teste.

        - **Retorna:**
        - `train_x` ou `test_x` (pd.DataFrame): DataFrame transformado.
        - `dic_transform` (dict): Dicionário de transformadores utilizados.
        """
        dados_trans_numerical = dados.copy()
        
        if modelo_imputer is None:
            dic_transform = {}
            for col in col_transformada:

                dados_trans_numerical[col], transformador_num = self.tipos_variaveis_numericas(dados_trans_numerical[col].values.reshape(-1, 1), tipo=tipo)

                dic_transform[col] = transformador_num

            return dados_trans_numerical, dic_transform
        else:
            dados_trans_numerical = dados.copy()
            for col in col_transformada:
                dados_trans_numerical[col] = modelo_imputer[col].transform(dados_trans_numerical[col].values.reshape(-1, 1))
            return dados_trans_numerical