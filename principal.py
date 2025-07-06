import os
import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

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
    transformacoes = ['mirror', 'rotate+10', 'rotate-10']
    idx = 0

    while len(imagens_classe) < TARGET_IMAGENS_POR_CLASSE:
        img = imagens_classe[idx % len(imagens_classe)]
        tipo = transformacoes[idx % len(transformacoes)]

        if tipo == 'mirror':
            nova = ImageOps.mirror(img)
        elif tipo == 'rotate+10':
            nova = img.rotate(10)
        elif tipo == 'rotate-10':
            nova = img.rotate(-10)

        imagens_classe.append(nova)
        idx += 1
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
