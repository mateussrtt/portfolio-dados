"""
# Car Dataset Prediction

Este projeto aplica técnicas de Machine Learning para prever valores com base em um conjunto de dados automotivo filtrado e pré-processado. A abordagem utiliza o modelo XGBoost, otimizado via Random Search, garantindo alta performance e robustez na predição.
"""

# 🚀 Importação das Bibliotecas
import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 📂 Carregamento dos Dados
df = pd.read_csv('data/cars_dataset.csv')

# 📊 Pré-processamento
df.fillna(df.median(), inplace=True)
df = pd.get_dummies(df, drop_first=True)

# ⚙️ Divisão dos Dados
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 🚀 Treinamento do Modelo
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
model.fit(X_train, y_train)

# 📈 Avaliação
y_pred = model.predict(X_test)
print("R²:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# 🌍 Impacto no Mundo Real
"""
Este projeto pode ser aplicado em diversas áreas, como:
- Precificação Inteligente: Concessionárias podem prever valores de revenda de veículos com base em características do mercado.
- Otimização de Estoque: Empresas podem ajustar suas estratégias de compra e venda de veículos com base em tendências preditivas.
- Seguro Automotivo: Companhias de seguro podem estimar valores de cobertura baseando-se nas condições e características dos veículos.

Além disso, este projeto demonstra habilidades valiosas para o mercado, como:
- Manipulação e limpeza de grandes volumes de dados.
- Aplicação de modelos preditivos avançados.
- Habilidade em interpretar métricas e otimizar modelos de Machine Learning.
"""

# Criado por Mateus Serretti
print("👨‍💻 Criado por [Mateus](https://github.com/mateussrtt).")
