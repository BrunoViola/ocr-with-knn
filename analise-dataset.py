import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# carregar os dados salvos
data = np.load('dataset_preprocessado.npz')
X_train = data['X_train']
y_train = data['y_train']
X_test  = data['X_test']
y_test  = data['y_test']

# ---------------------------------------------------------
# Distribuição geral
y_total = np.concatenate([y_train, y_test])

# ---------------------------------------------------------
# Gráfico treinamento: quantidade de imagens em cada classe apenas no conjunto de treinamento.
plt.figure(figsize=(10,5))
sns.countplot(x=y_train, palette="Blues")
plt.title("Distribuição de classes - Treinamento")
plt.xlabel("Classe")
plt.ylabel("Quantidade")
plt.show()

# ---------------------------------------------------------
# Gráfico teste: quantidade de imagens em cada classe apenas no conjunto de teste.
plt.figure(figsize=(10,5))
sns.countplot(x=y_test, palette="Greens")
plt.title("Distribuição de classes - Teste")
plt.xlabel("Classe")
plt.ylabel("Quantidade")
plt.show()

# ---------------------------------------------------------
# Gráfico geral: quantidade de imagens em cada classe apenas nos conjuntos de teste e treino (tudo).
plt.figure(figsize=(10,5))
sns.countplot(x=y_total, palette="Purples")
plt.title("Distribuição de classes - Geral")
plt.xlabel("Classe")
plt.ylabel("Quantidade")
plt.show()
