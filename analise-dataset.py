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

# Cálculo do maior valor entre todas as classes (para fixar o eixo Y)
from collections import Counter
contagem_total = Counter(y_total)
y_max = max(contagem_total.values())  # maior quantidade entre as classes

# ---------------------------------------------------------
# Gráfico treinamento
plt.figure(figsize=(10,5))
sns.countplot(x=y_train, palette="Blues")
plt.ylim(0, y_max)  # fixa o limite Y
plt.title("Distribuição de classes - Treinamento")
plt.xlabel("Classe")
plt.ylabel("Quantidade")
plt.show()

# ---------------------------------------------------------
# Gráfico teste
plt.figure(figsize=(10,5))
sns.countplot(x=y_test, palette="Greens")
plt.ylim(0, y_max)  # fixa o limite Y
plt.title("Distribuição de classes - Teste")
plt.xlabel("Classe")
plt.ylabel("Quantidade")
plt.show()

# ---------------------------------------------------------
# Gráfico geral
plt.figure(figsize=(10,5))
sns.countplot(x=y_total, palette="Purples")
plt.ylim(0, y_max)  # fixa o limite Y
plt.title("Distribuição de classes - Geral")
plt.xlabel("Classe")
plt.ylabel("Quantidade")
plt.show()
