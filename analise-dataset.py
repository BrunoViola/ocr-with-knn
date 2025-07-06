import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

#carrega os dados salvos no .npz
data = np.load('dataset_preprocessado.npz')
X_train = data['X_train']
y_train = data['y_train']
X_test  = data['X_test']
y_test  = data['y_test']

# ---------------------------------------------------------
#concatena os dados de treino e teste para análise geral
y_total = np.concatenate([y_train, y_test])

#cálculo do maior valor entre todas as classes para fixar o eixo Y
contagem_total = Counter(y_total)
y_max = max(contagem_total.values())  #maior quantidade entre as classes

# ---------------------------------------------------------
#gráfico da distribuição no treinamento
plt.figure(figsize=(10,5))
sns.countplot(x=y_train, palette="Blues")
plt.ylim(0, y_max)  # fixa o limite Y
plt.title("Distribuição de classes - Treinamento")
plt.xlabel("Classe")
plt.ylabel("Quantidade")
plt.show()

# ---------------------------------------------------------
# gráfico da distribuição no teste
plt.figure(figsize=(10,5))
sns.countplot(x=y_test, palette="Greens")
plt.ylim(0, y_max)  # fixa o limite Y
plt.title("Distribuição de classes - Teste")
plt.xlabel("Classe")
plt.ylabel("Quantidade")
plt.show()

# ---------------------------------------------------------
# gráfico da distribuição geral
plt.figure(figsize=(10,5))
sns.countplot(x=y_total, palette="Purples")
plt.ylim(0, y_max)  # fixa o limite Y
plt.title("Distribuição de classes - Geral")
plt.xlabel("Classe")
plt.ylabel("Quantidade")
plt.show()
