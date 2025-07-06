import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns

#carrega o dataset pré-processado
data = np.load('dataset_preprocessado.npz')
X_train_total = data['X_train']
y_train_total = data['y_train']
X_test_total = data['X_test']
y_test_total = data['y_test']

#junta tudo para reembaralhar a cada iteração
X_total = np.concatenate((X_train_total, X_test_total), axis=0)
y_total = np.concatenate((y_train_total, y_test_total), axis=0)

#parametros do k-NN
k = 3
num_iteracoes = 10

acuracias = []
precisoes = []
revocoes = []
f1_scores = []
matrizes_confusao = []

print(f"Iniciando treino e teste com k-NN (k = {k})...\n")

for i in range(num_iteracoes):
    print(f"--- Iteração {i+1} ---")
    
    X_train, X_test, y_train, y_test = train_test_split(X_total, y_total, test_size=0.2, stratify=y_total, random_state=i) # utiliza i no random_state para garantir reprodutibilidade 

    # achata as imagens
    X_train_flat = X_train.reshape((X_train.shape[0], -1))
    X_test_flat = X_test.reshape((X_test.shape[0], -1))

    # treinamento do knn
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_flat, y_train)

    y_pred = knn.predict(X_test_flat)

    # cálculo das metricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    acuracias.append(acc)
    precisoes.append(prec)
    revocoes.append(rec)
    f1_scores.append(f1)

    # matriz de confusão
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(25))
    matrizes_confusao.append(cm)

    print(f"Acurácia : {acc:.4f}")
    print(f"Precisão : {prec:.4f}")
    print(f"Revocação: {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print()

#exibição das médias das métricas
print("=== MÉDIAS APÓS 10 ITERAÇÕES ===")
print(f"Acurácia média : {np.mean(acuracias):.4f}")
print(f"Precisão média : {np.mean(precisoes):.4f}")
print(f"Revocação média: {np.mean(revocoes):.4f}")
print(f"F1-Score médio : {np.mean(f1_scores):.4f}")

#soma das matrizes elemento por elemento para ver o padrão geral
cm_total = np.sum(matrizes_confusao, axis=0)

#plot da matriz de confusão final
plt.figure(figsize=(12, 10))
sns.heatmap(cm_total, annot=True, fmt='d', cmap='Blues')
plt.title("Matriz de Confusão - Soma das 10 Iterações")
plt.xlabel("Classe Predita")
plt.ylabel("Classe Verdadeira")
plt.tight_layout()
plt.show()
