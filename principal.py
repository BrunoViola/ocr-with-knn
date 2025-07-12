import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import src.pre_processamento as pre_processamento
from src.variaveis import dimensao_imagem

# caminho para os diretórios das classes (0 a 24)
base_dir = 'dataset'
img_size = (dimensao_imagem, dimensao_imagem)  # tamanho padronizado pela var dimensao_imagem

X = []  # lista para armazenar as imagens pré-processadas
y = []  # lista para armazenar os rótulos

# percorre as 25 classes (0 a 24)
for label in range(25):
    label_dir = os.path.join(base_dir, str(label))  # caminho da pasta de cada classe
    imagens_classe = []  # armazena as imagens da classe atual

    for filename in os.listdir(label_dir):
        if filename.endswith(('.jpeg', '.jpg')):
            img_path = os.path.join(label_dir, filename)
            img = Image.open(img_path)
            img = pre_processamento.pre_processar_imagem(img, dimensao_imagem)  # pré-processa a imagem
            imagens_classe.append(img)

    for img in imagens_classe:
        img_array = np.array(img)
        X.append(img_array)
        y.append(label)

# conversão das listas para numpy arrays
X = np.array(X).reshape(-1, dimensao_imagem, dimensao_imagem, 1)
y = np.array(y)

# salva os arrays para uso posterior no classificacao_knn.py
np.savez('dataset_preprocessado.npz', X_total=X, y_total=y)

# ----- exibição de informações sobre o dataset -----
# divisão treino/teste (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# exibe as quantidades
print(f"Imagens carregadas: {len(X)}")
print(f"Treinamento: {len(X_train)} | Teste: {len(X_test)}")

# distribuição geral
classes, counts = np.unique(y, return_counts=True)
print("Distribuição de imagens por classe (geral):")
for cls, count in zip(classes, counts):
    print(f"Classe {cls}: {count} imagens")

print("\nDistribuição no conjunto de treinamento:")
classes, counts = np.unique(y_train, return_counts=True)
for cls, count in zip(classes, counts):
    print(f"Classe {cls}: {count} imagens")

print("\nDistribuição no conjunto de teste:")
classes, counts = np.unique(y_test, return_counts=True)
for cls, count in zip(classes, counts):
    print(f"Classe {cls}: {count} imagens")
# -----------------------------------------------------

# executa a classificação KNN
import src.classificacao_knn as classificacao_knn # executando a classificação KNN