import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# Caminho para os diretórios das classes (0 a 24)
base_dir = 'dataset'
img_size = (32, 32)  # Tamanho padronizado

X = []
y = []

# Percorre cada classe (0 a 24)
for label in range(25):
    label_dir = os.path.join(base_dir, str(label))
    for filename in os.listdir(label_dir):
        if filename.endswith('.jpeg'):
            img_path = os.path.join(label_dir, filename)
            img = Image.open(img_path).convert('L')         # Converte para grayscale
            img = img.resize(img_size)                      # Redimensiona
            img_array = np.array(img) / 255.0               # Normaliza entre 0 e 1
            X.append(img_array)
            y.append(label)

# Converte para numpy arrays
X = np.array(X).reshape(-1, 32, 32, 1)  # Adiciona canal único
y = np.array(y)

# Divide entre treino (80%) e teste (20%) com embaralhamento
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Salva os arrays para uso posterior
np.savez('dataset_preprocessado.npz', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

print(f"Imagens carregadas: {len(X)}")
print(f"Treinamento: {len(X_train)} | Teste: {len(X_test)}")
