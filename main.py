# importar as bibliotecas para o sistema
# biblioteca de inteligencia artificial

from sklearn.linear_model import LinearRegression
# para instalar todas as bibliotecas necessarias voce deve utilizar os comandos: pip install pandas matplotlib scikit-learn ou py -m pip install pandas matplotlib scikit-learn (no mac utilize pip3)

# a inteligencia artificial em geral nao trabalha sozinha, ela depende de ferramentas para tratar os dados

import pandas as pd
# pandas para ler os dados

import matplotlib.pyplot as plt
# para mostrar os dados

# entrada de dados - ler os dados
# processamento de dados - interpretar eles
# saida de dados - exibicao de informacoes

# 1 - LER
df_dados = pd.read_csv("dados.csv")
print(df_dados) #exibi tudo
print(df_dados.head()) # exibi 5 linhas

# 2 - PROCESSAR OS DADOS
# as notas dependem da quantidade de horas estudadas, quanto mais horas estudadas maior a nota.
# variaveis independentes - horas de estudo
x_independente = df_dados[["horas_estudo"]]

# variaveis dependentes - nota
y_dependente = df_dados[["nota"]]

# 2.1- criar um modelo de regressao linear
modelo = LinearRegression()

# 2.2 - trinar o model (PRIMEIRO A VARIAVEL INDEPENDENTE)
modelo.fit(x_independente, y_dependente)

# 2.3 - exibir os dados
print("coeficiente", modelo.coef_[0]) # inclinacao
print("interpretacao", modelo.intercept_) # onde os pontos se encontram

# 3 - SAIDA DOS DADOS
# 3.1 - o que eu quero prever?
nova_hora = [[float(input("quantas horas voce pretende estudar? "))]] # anota o que quer prever
print(nova_hora)

# vou prever
prever = modelo.predict(nova_hora)

# mostrar a previsao
print(f"se voce estudar {nova_hora} sua nota vai ser de {prever}")

# SEMPRE A REGRESSAO LINEAR OU SEJA A PREVISAO DEVE TER DOIS GRAFICOS EM UM
# SENDO ELES O GRAFICO DE DISPERSAO COM O GRAFICO DE LINHA
plt.plot(df_dados["horas_estudo"], modelo.predict(x_independente))
plt.scatter(df_dados["horas_estudo"], df_dados["nota"])

plt.show()