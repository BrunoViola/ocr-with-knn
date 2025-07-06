import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

#caminho para os diretórios das classes (0 a 24)
base_dir = 'dataset'
img_size = (32, 32)  #tamanho padronizado (32x32)

X = [] # lista para armazenar as imagens pré-processadas
y = [] # lista para armazenar os rótulos

# percorre as 25 classes (0 a 24)
for label in range(25):
    label_dir = os.path.join(base_dir, str(label)) # caminho da pasta de cada classe
    #percorre cada imagem na pasta da classe
    for filename in os.listdir(label_dir):
        if filename.endswith(('.jpeg','.jpg')): #verifica se o arquivo é uma imagem JPEG ou JPG
            img_path = os.path.join(label_dir, filename) #caminho da imagem
            img = Image.open(img_path).convert('L')         #converte para escala de cinza
            img = img.resize(img_size)                      #redimensiona
            X.append(img)                                   #adiciona a imagem à lista
            y.append(label)                                 #adiciona o rótulo correspondente à lista

#conversão das duas listas para numpy arrays
X = np.array(X).reshape(-1, 32, 32, 1)
y = np.array(y)

#divisão do dataset em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#salva os arrays para uso posterior
np.savez('dataset_preprocessado.npz', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

#exibe as quantidades de imagens do dataset
print(f"Imagens carregadas: {len(X)}")
print(f"Treinamento: {len(X_train)} | Teste: {len(X_test)}")
# -------------------------------------------------------

# distribuição de imagens por classe no dataset completo
classes, counts = np.unique(y, return_counts=True)
print("Distribuição de imagens por classe (geral):")
for cls, count in zip(classes, counts):
    print(f"Classe {cls}: {count} imagens")
# -------------------------------------------------------

#distribuição de imagens por classe nos conjuntos de treino e teste separados
print("\nDistribuição no conjunto de treinamento:")
classes, counts = np.unique(y_train, return_counts=True)
for cls, count in zip(classes, counts):
    print(f"Classe {cls}: {count} imagens")

print("\nDistribuição no conjunto de teste:")
classes, counts = np.unique(y_test, return_counts=True)
for cls, count in zip(classes, counts):
    print(f"Classe {cls}: {count} imagens")
# --------------------------------------------------------------------------
