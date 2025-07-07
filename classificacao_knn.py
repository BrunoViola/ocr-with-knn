import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report)
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageEnhance
import random

import graficos_knn

# parâmetros
k = 5
num_iteracoes = 10
TARGET_POR_CLASSE = 300

data_augmentation_flag = 0 # flag para identificar se o data augmentation está sendo realizado
dimensao_imagem = 32

relatorio_por_classe = defaultdict(lambda: {'precision': [], 'recall': [], 'f1-score': [], 'support': 0}) # dicionário para armazenar as métricas por classe para exibição do classification report final

# carrega o dataset pré-processado
data = np.load('dataset_preprocessado.npz')
X_train_total = data['X_train']
y_train_total = data['y_train']
X_test_total = data['X_test']
y_test_total = data['y_test']

# junta tudo para reembaralhar nas iterações
X_total = np.concatenate((X_train_total, X_test_total), axis=0)
y_total = np.concatenate((y_train_total, y_test_total), axis=0)

# inicializa listas para armazenar resultados
acuracias = []
precisoes = []
revocoes = []
f1_scores = []
matrizes_confusao = []

print(f"Iniciando treino e teste com k-NN (k = {k})...\n")

for i in range(num_iteracoes):
    print(f"--- Iteração {i+1} ---")
    
    #divide os dados
    X_train, X_test, y_train, y_test = train_test_split(
        X_total, y_total, test_size=0.2, stratify=y_total, random_state=i
    )

    # Data augmentation somente no conjunto de treino
    if data_augmentation_flag:
        X_train_aug = []
        y_train_aug = []

        for label in np.unique(y_train): # itera sobre cada classe
            imagens_classe = [
                Image.fromarray((img.reshape(dimensao_imagem, dimensao_imagem)).astype(np.uint8)) # converte para imagem
                for img, y in zip(X_train, y_train) if y == label # filtra imagens da classe atual
            ]
            
            while len(imagens_classe) < TARGET_POR_CLASSE: # enquanto não atingir o target
                img_base = random.choice(imagens_classe)
                nova = img_base.copy()
                tipo = random.randint(1, 5)

                if tipo == 1:
                    nova = img_base.rotate(random.randint(-10, 10))
                elif tipo == 2:
                    enhancer = ImageEnhance.Brightness(img_base)
                    nova = enhancer.enhance(random.uniform(0.5, 1.5))
                elif tipo == 3:
                    enhancer = ImageEnhance.Contrast(img_base)
                    nova = enhancer.enhance(random.uniform(0.5, 1.5))
                elif tipo == 4:
                    enhancer = ImageEnhance.Sharpness(img_base)
                    nova = enhancer.enhance(random.uniform(0.5, 1.5))

                imagens_classe.append(nova)

            for img in imagens_classe:
                img_np = np.array(img)
                X_train_aug.append(img_np.reshape(dimensao_imagem, dimensao_imagem, 1)) # mantém a dimensão original
                y_train_aug.append(label)

        X_train = np.array(X_train_aug)
        y_train = np.array(y_train_aug)
    
    # -------------------------------------------------------

    # achata as imagens para usar no KNN
    X_train_flat = X_train.reshape((X_train.shape[0], -1))
    X_test_flat = X_test.reshape((X_test.shape[0], -1))
    
    #treino do KNN
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_flat, y_train)

    y_pred = knn.predict(X_test_flat)

    #métricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    acuracias.append(acc)
    precisoes.append(prec)
    revocoes.append(rec)
    f1_scores.append(f1)

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(25))
    matrizes_confusao.append(cm)

    print(f"Acurácia : {acc:.4f}")
    print(f"Precisão : {prec:.4f}")
    print(f"Revocação: {rec:.4f}")
    print(f"F1-Score : {f1:.4f}\n")

    # construção do relatório de classificação
    report = classification_report(y_test, y_pred, labels=np.arange(25), output_dict=True, zero_division=0)

    for classe in map(str, range(25)):
        if classe in report:
            relatorio_por_classe[classe]['precision'].append(report[classe]['precision'])
            relatorio_por_classe[classe]['recall'].append(report[classe]['recall'])
            relatorio_por_classe[classe]['f1-score'].append(report[classe]['f1-score'])
            relatorio_por_classe[classe]['support'] += report[classe]['support']
    # -------------------------------------------------------

#resultados finais
print("=== MÉDIAS APÓS 10 ITERAÇÕES ===")
print(f"Acurácia média : {np.mean(acuracias):.4f}")
print(f"Precisão média : {np.mean(precisoes):.4f}")
print(f"Revocação média: {np.mean(revocoes):.4f}")
print(f"F1-Score médio : {np.mean(f1_scores):.4f}")

# cálculo da média do relatório de classificação e salva em arquivo txt
arquivo_txt_relatorio = open('relatorio_classificacao.txt', 'w')
print("=== MEDIA DAS METRICAS POR CLASSE (10 iteracoes) ===", file=arquivo_txt_relatorio)
print("========== Dimensao das imagens:", dimensao_imagem, "x", dimensao_imagem, "==========" , file=arquivo_txt_relatorio)
print("=============== Com Data Augmentation no Treino ==============" if data_augmentation_flag else "=============== Sem Data Augmentation ==============", file=arquivo_txt_relatorio)
print(f"{'Classe':7} | {'Precisao':9} | {'Revocacao':9} | {'F1-Score':9} | {'Suporte':9}", file=arquivo_txt_relatorio)
for classe in sorted(relatorio_por_classe.keys(), key=int):
    p = np.mean(relatorio_por_classe[classe]['precision'])
    r = np.mean(relatorio_por_classe[classe]['recall'])
    f = np.mean(relatorio_por_classe[classe]['f1-score'])
    s = relatorio_por_classe[classe]['support']
    print(f"{classe:<7} | {p:<9.2f} | {r:<9.2f} | {f:<9.2f} | {s:<9.0f}", file=arquivo_txt_relatorio)
# -----------------------------------------------------------------------

#matriz de confusão somada
cm_total = np.sum(matrizes_confusao, axis=0)

#plot da matriz de confusão
plt.figure(figsize=(12, 10))
sns.heatmap(cm_total, annot=True, fmt='d', cmap='Blues')
if data_augmentation_flag:
    plt.title(f"Matriz de Confusão - Soma das 10 Iterações (Com Data Augmentation no Treino)\nDimensão das imagens {dimensao_imagem}x{dimensao_imagem}")
else:
    plt.title(f"Matriz de Confusão - Soma das 10 Iterações (Sem Data Augmentation)\nDimensão das imagens {dimensao_imagem}x{dimensao_imagem}")
plt.xlabel("Classe Predita")
plt.ylabel("Classe Verdadeira")
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()

graficos_knn.plotar_metricas_por_classe(relatorio_por_classe)
