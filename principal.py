import os
import numpy as np
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split
import random

# caminho para os diretórios das classes (0 a 24)
base_dir = 'dataset'
img_size = (32, 32)  # tamanho padronizado (32x32)
TARGET_IMAGENS_POR_CLASSE = 300  # === CÓDIGO NOVO: meta de imagens por classe

X = []  # lista para armazenar as imagens pré-processadas
y = []  # lista para armazenar os rótulos

# percorre as 25 classes (0 a 24)
for label in range(25):
    label_dir = os.path.join(base_dir, str(label))  # caminho da pasta de cada classe
    imagens_classe = []  # armazena as imagens da classe atual

    for filename in os.listdir(label_dir):
        if filename.endswith(('.jpeg', '.jpg')):
            img_path = os.path.join(label_dir, filename)
            img = Image.open(img_path).convert('L')
            img = img.resize(img_size)
            imagens_classe.append(img)

    # Geração de imagens adicionais para balancear o dataset
    '''
    while len(imagens_classe) < TARGET_IMAGENS_POR_CLASSE:
        img_base = random.choice(imagens_classe)  # escolhe uma imagem aleatória da classe
        variacao = random.randint(1, 5)  # número de variações a serem criadas
        nova = img_base.copy()  # cria uma cópia da imagem base

        if variacao == 1:
            nova = img_base.rotate(random.randint(-10, 10))
        elif variacao == 2:
            enhancer = ImageEnhance.Brightness(img_base) # ajuste de brilho
            nova = enhancer.enhance(random.uniform(0.5, 1.5))
        elif variacao == 3:
            enhancer = ImageEnhance.Contrast(img_base) # ajuste de contraste
            nova = enhancer.enhance(random.uniform(0.5, 1.5))
        elif variacao == 4:
            enhancer = ImageEnhance.Sharpness(img_base) # ajuste de nitidez
            nova = enhancer.enhance(random.uniform(0.5, 1.5))

        imagens_classe.append(nova)
    '''
    # -------------------------------------------------------

    for img in imagens_classe:
        img_array = np.array(img)
        X.append(img_array)
        y.append(label)

# conversão das listas para numpy arrays
X = np.array(X).reshape(-1, 32, 32, 1)
y = np.array(y)

# divisão treino/teste (80/20) com estratificação
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# salva os arrays para uso posterior
np.savez('dataset_preprocessado.npz', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

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
