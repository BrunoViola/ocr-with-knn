import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -------------------------------
# 1. Carregando os dados salvos
# -------------------------------
# Carrega os arrays salvos no dataset_preprocessado.npz
data = np.load('dataset_preprocessado.npz')
X_train_total = data['X_train']
y_train_total = data['y_train']
X_test_total = data['X_test']
y_test_total = data['y_test']

# Junta tudo para sortear aleatoriamente nas 10 iterações
X_total = np.concatenate((X_train_total, X_test_total), axis=0)
y_total = np.concatenate((y_train_total, y_test_total), axis=0)

# -------------------------------
# 2. Parâmetros do modelo e das iterações
# -------------------------------
k = 3  # Número de vizinhos para o KNN
num_iteracoes = 10  # Total de rodadas para teste

# Armazena os resultados de cada rodada
acuracias = []
precisoes = []
revocoes = []
f1_scores = []

print("Iniciando treino e teste com k-NN (k = 3)...\n")

# -------------------------------
# 3. Loop principal das iterações
# -------------------------------
for i in range(num_iteracoes):
    print(f"--- Iteração {i+1} ---")

    # Divide aleatoriamente o conjunto total em treino e teste (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_total, y_total, test_size=0.2, stratify=y_total, random_state=i
    )

    # Redimensiona as imagens de 32x32x1 para vetores 1D (flatten) para o k-NN
    X_train_flat = X_train.reshape((X_train.shape[0], -1))
    X_test_flat = X_test.reshape((X_test.shape[0], -1))

    # Cria e treina o modelo k-NN
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_flat, y_train)

    # Realiza as previsões no conjunto de teste
    y_pred = knn.predict(X_test_flat)

    # Calcula as métricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    # Armazena os resultados
    acuracias.append(acc)
    precisoes.append(prec)
    revocoes.append(rec)
    f1_scores.append(f1)

    # Exibe as métricas da iteração atual
    print(f"Acurácia : {acc:.4f}")
    print(f"Precisão : {prec:.4f}")
    print(f"Revocação: {rec:.4f}")
    print(f"F1-Score : {f1:.4f}\n")

# -------------------------------
# 4. Exibir as médias finais
# -------------------------------
print("=== MÉDIAS APÓS 10 ITERAÇÕES ===")
print(f"Acurácia média : {np.mean(acuracias):.4f}")
print(f"Precisão média : {np.mean(precisoes):.4f}")
print(f"Revocação média: {np.mean(revocoes):.4f}")
print(f"F1-Score médio : {np.mean(f1_scores):.4f}")
