# Regression-of-Used-Car-Prices

## Objetivo

O objetivo desta competição é prever o preço de carros usados com base em vários atributos. Você deve criar um modelo que estime o preço de um carro dado um conjunto de características. O link para o conjunto de dados é [link](https://www.kaggle.com/competitions/playground-series-s4e9/overview)

## Avaliação
As submissões serão avaliadas com base na Raiz do Erro Quadrático Médio (RMSE). A fórmula para calcular o RMSE é a seguinte:


$$
\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}
$$

Onde:

- `ŷ_i` é o valor previsto para a i-ésima instância
- `y_i` é o valor real para a i-ésima instância
- `N` é o número total de instâncias


## Arquivo de Submissão
Para cada ID no conjunto de teste, você deve prever o preço do carro. O arquivo de submissão deve conter um cabeçalho e seguir o formato abaixo:

```text
id,price
188533,43878.016
188534,43878.016
188535,43878.016
etc.
```

## Etapas

### Tratamento de Dados

### Principais Características

Utilizando a biblioteca [Shapley Additive exPlanations (SHAP)](https://shap.readthedocs.io/en/latest/), podemos identificar as variáveis mais importantes no modelo de regressão, com destaque para as variáveis `milage`, `hp`, `idade_carro`, `brand_Porsche`, e `brand_Land`.

Ao analisar o gráfico de `summary_plot`, observamos os seguintes insights:

- **`milage` (quilometragem)**: Valores **baixos** de quilometragem (em azul) têm um impacto **positivo** nas previsões, enquanto valores **altos** (em vermelho) impactam **negativamente**. Isso indica que carros com quilometragem mais alta tendem a ter preços **mais baixos**, enquanto veículos com baixa quilometragem são avaliados com preços **mais altos**.

- **`hp` (potência)**: Valores **altos** de potência estão associados a impactos **positivos** nas previsões, sugerindo que carros com mais potência são, em geral, previstos com preços **mais altos**.

- **`idade_carro` (idade do carro)**: Carros mais antigos (em vermelho) têm um impacto **negativo** no valor previsto, enquanto carros mais novos (em azul) têm um efeito **positivo**, o que é esperado, já que veículos mais novos geralmente possuem maior valor de mercado.

- **`brand_Porsche`**: A marca Porsche tem um impacto **fortemente positivo** nas previsões, indicando que os carros dessa marca são frequentemente associados a preços **mais elevados**, como esperado, dado o prestígio e o alto custo dos veículos Porsche.

- **`cambio_manual`**: Veículos com câmbio manual parecem ter um impacto **negativo** nas previsões, sugerindo que carros com câmbio manual são frequentemente avaliados com preços **mais baixos** em comparação com aqueles com câmbio automático ou dual.

- **`accident_At least 1 accident or damage reported` (acidente/dano reportado)**: Como esperado, a presença de acidentes ou danos reportados (valores em vermelho) tem um impacto **negativo** significativo, diminuindo o valor previsto do carro. Isso reflete a percepção de que carros com histórico de acidentes ou danos tendem a ser desvalorizados.

Este gráfico ajuda a visualizar o efeito de cada variável no modelo, ilustrando como diferentes características afetam o preço previsto de um veículo.

![Analise Exploratória](imagens/shap_summary_plot.png)